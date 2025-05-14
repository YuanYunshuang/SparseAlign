from typing import List

import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.modules.utils.misc import SELayer_Linear, MLN
from cosense3d.modules.utils.positional_encoding import pos2posemb3d


class PETR(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 position_range,
                 num_reg_fcs=2,
                 num_pred=3,
                 topk=2048,
                 num_query=644,
                 depth_num=64,
                 LID=True,
                 depth_start=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.img_position_dim = depth_num * 3
        self.num_pose_feat = 64
        self.in_channels = in_channels
        self.topk = topk
        self.num_query = num_query
        self.LID = LID
        self.num_reg_fcs = num_reg_fcs
        self.num_pred = num_pred

        if self.LID: # linear-increasing discretization
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (position_range[3] - depth_start) / (depth_num * (1 + depth_num))
            coords_d = depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=depth_num, step=1).float()
            bin_size = (position_range[3] - depth_start) / depth_num
            coords_d = depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)
        self.position_range = nn.Parameter(torch.tensor(position_range), requires_grad=False)
        self.reference_points = nn.Embedding(self.num_query, 3)

        self._init_layers()

    def _init_layers(self):
        self.img_position_encoder = nn.Sequential(
            nn.Linear(self.img_position_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.img_memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat*3, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(8, f_dim=self.embed_dims)
        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, img_feat, img_roi, img_coor, img_size, intrinsics, lidar2img, **kwargs):
        img_memory, img_pos, img2lidars, Is = self.gather_topk(
            img_feat, img_roi, img_coor, img_size, intrinsics, lidar2img)

        img_pos_emb, cone = self.img_position_embeding(img_memory, img_pos, Is, img2lidars)
        img_memory = self.img_memory_embed(img_memory)

        # spatial_alignment in focal petr
        img_memory = self.spatial_alignment(img_memory, cone)
        img_pos_emb = self.featurized_pe(img_pos_emb, img_memory)

        reference_points = (self.reference_points.weight).unsqueeze(0).repeat(img_memory.shape[0], 1, 1)
        query_pos = self.query_embedding(pos2posemb3d(reference_points, self.num_pose_feat))
        tgt = torch.zeros_like(query_pos)
        outs_dec, _ = self.transformer(img_memory, tgt, query_pos, img_pos_emb)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(img_memory))
        ]

        return {self.scatter_keys[0]: outs}

    def format_input(self, input: List):
        memory = []
        for x in input:
            x = x.permute(0, 2, 3, 1).flatten(0, 2)
            memory.append(x)
        max_l = max([m.shape[0] for m in memory])
        out = x.new_zeros(len(memory), max_l, x.shape[-1])
        mask = x.new_ones(len(memory), max_l)
        for i, m in enumerate(memory):
            out[i, :len(m)] = m
            mask[i, :len(m)] = False
        return out, mask

    def gather_topk(self, img_feat, img_roi, img_coor, img_size, intrinsics, lidar2img):
        B = len(img_feat)
        out_feat = []
        out_coor = []
        mem_ctrs = []
        img2lidars = []
        Is = []
        for b in range(B):
            topk_inds = img_roi[b]['sample_weight'].view(-1).topk(k=self.topk).indices
            out_feat.append(img_feat[b].permute(0, 2, 3, 1).flatten(0, 2)[topk_inds])
            # out_coor.append(img_coor[b].flatten(0, 2)[topk_inds])
            N, _, h, w = img_feat[b].shape
            H, W = img_size[b][0]

            # [alpha_x, alpha_y]
            intrinsic = torch.stack(intrinsics[b], dim=0)[..., [0, 1], [0, 1]]
            intrinsic = torch.abs(intrinsic) / 1e3
            intrinsic = intrinsic.view(N, -1, 2).repeat(1, h * w, 1).flatten(0, 1)[topk_inds]
            Is.append(intrinsic)

            # transform memery_centers from ratio to pixel
            img_coor[b][..., 0] = img_coor[b][..., 0] * W
            img_coor[b][..., 1] = img_coor[b][..., 1] * H
            topk_ctrs = img_coor[b].flatten(0, 2)[topk_inds]
            mem_ctrs.append(topk_ctrs)

            img2lidar = torch.stack(lidar2img[b], dim=0).inverse()
            img2lidar = img2lidar.view(N, 1, 4, 4).repeat(1, h * w, 1, 1)
            img2lidars.append(img2lidar.flatten(0, 1)[topk_inds])

        out_feat = torch.stack(out_feat, dim=0)
        # out_coor = torch.stack(out_coor, dim=0)
        mem_ctrs = torch.stack(mem_ctrs, dim=0)
        img2lidars = torch.stack(img2lidars, dim=0)
        Is = torch.stack(Is, dim=0)

        return out_feat, mem_ctrs, img2lidars, Is

    def img_position_embeding(self, img_memory, img_pos, Is, img2lidars):
        eps = 1e-5
        B = len(img_memory)
        D = self.coords_d.shape[0]
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, self.topk, 1, 1)
        img_pos = img_pos.unsqueeze(-2).repeat(1, 1, D, 1)
        coords = torch.cat([img_pos, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords_d)), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], torch.ones_like(coords_d) * eps)
        coords = coords.unsqueeze(-1)

        coords3d = torch.matmul(img2lidars.unsqueeze(-3), coords).squeeze(-1)[..., :3]
        coords3d[..., :3] = (coords3d[..., :3] - self.position_range[:3]) / (
                self.position_range[3:] - self.position_range[:3])
        coords3d = coords3d.reshape(B, -1, D * 3)
        pos_embed = inverse_sigmoid(coords3d)
        coords_position_embeding = self.img_position_encoder(pos_embed)
        cone = torch.cat([Is, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)
        return coords_position_embeding, cone


