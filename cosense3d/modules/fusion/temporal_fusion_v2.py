from typing import Mapping, Any

import torch
import torch.nn as nn
from torch_scatter import scatter_max

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.misc import MLN
from cosense3d.modules.utils.common import cat_coor_with_idx
import cosense3d.modules.utils.positional_encoding as PE
from cosense3d.modules.plugin.obj_flow import ObjFlow


class LocalTemporalFusion(BaseModule):
    """Modified from TemporalFusion to standardize input and output keys"""
    def __init__(self,
                 in_channels,
                 transformer,
                 lidar_range,
                 pos_dim=3,
                 num_pose_feat=128,
                 topk_ref_pts=1024,
                 topk_feat=512,
                 num_propagated=256,
                 memory_len=1024,
                 ref_pts_stride=2,
                 transformer_itrs=1,
                 global_ref_time=0,
                 norm_fusion=False,
                 obj_flow_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        if obj_flow_layer is not None:
            self.obj_flow_layer = plugin.build_plugin_module(obj_flow_layer)
        else:
            self.obj_flow_layer = None
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.topk_ref_pts = topk_ref_pts
        self.topk_feat = topk_feat
        self.ref_pts_stride = ref_pts_stride
        self.num_propagated = num_propagated
        self.memory_len = memory_len
        self.transformer_itrs = transformer_itrs
        self.global_ref_time = global_ref_time
        self.norm_fusion = norm_fusion

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        pose_nerf_dim = (3 + 3 * 4) * 12
        self.ego_pose_pe = MLN(pose_nerf_dim, f_dim=self.embed_dims)
        self.ego_pose_memory = MLN(pose_nerf_dim, f_dim=self.embed_dims)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, local_roi, bev_feat, mem_dict, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(local_roi, bev_feat, self.ref_pts_stride, self.topk_ref_pts)

        ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))

        ref_time = None
        reference_points = ref_pos.clone()
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos, ext_feat = \
            self.temporal_alignment(query_pos, tgt, reference_points,
                                    ref_feat, mem_dict, ref_time)
        global_feat = []

        for _ in range(self.transformer_itrs):
            tgt = self.transformer(tgt, query_pos,
                                   temp_memory=temp_memory,
                                   temp_pos=temp_pos)[-1]
            global_feat.append(tgt)
        global_feat = torch.stack(global_feat, dim=0)
        local_feat = torch.cat([ref_feat, ext_feat], dim=1)
        local_feat = local_feat[None].repeat(self.transformer_itrs, 1, 1, 1)
        if self.norm_fusion:
            outs_dec = self.local_global_fusion(torch.cat([local_feat, global_feat], dim=-1))
        else:
            # simple addition will lead to large values in long sequences
            outs_dec = local_feat + global_feat

        ref = reference_points * (self.lidar_range[3:] - self.lidar_range[:3]) + self.lidar_range[:3]
        if self.obj_flow_layer is not None:
            C, F = self.obj_flow_layer(ref, outs_dec[-1])
        else:
            C = cat_coor_with_idx(ref)
            F = outs_dec[-1].flatten(0, 1)
        outs = []
        for i in range(len(bev_feat)):
            mask = C[:, 0] == i
            outs.append({
                    'outs_dec': F[mask].unsqueeze(1),
                    'ref_pts': C[mask, 1:],
                })
        return {self.scatter_keys[0]: outs}

    def gather_topk(self, rois, bev_feats, stride, topk):
        topk_feat, topk_ctr = [], []
        for roi, bev_feat in zip(rois, bev_feats):
            ctr = bev_feat[f'p{stride}']['ctr']
            feat = bev_feat[f'p{stride}']['feat']
            if 'scr' in roi:
                scores = roi['scr']
            else:
                scores = roi[f'p{stride}']['scr']
            sort_inds = scores.argsort(descending=True)
            if scores.shape[0] < topk:
                n_repeat = topk // len(scores) + 1
                sort_inds = torch.cat([sort_inds] * n_repeat, dim=0)

            topk_inds = sort_inds[:topk]
            topk_ctr.append(ctr[topk_inds])
            topk_feat.append(feat[topk_inds])
        topk_ctr = torch.stack(topk_ctr, dim=0)
        topk_feat = torch.stack(topk_feat, dim=0)
        # pad 2d coordinates to 3d if needed
        if topk_ctr.shape[-1] < self.pos_dim:
            pad_dim = self.pos_dim - topk_ctr.shape[-1]
            topk_ctr = torch.cat([topk_ctr, torch.zeros_like(topk_ctr[..., :pad_dim])], dim=-1)
        return topk_feat, topk_ctr

    def embed_pos(self, pos, dim=None):
        dim = self.num_pose_feat if dim is None else dim
        return getattr(PE, f'pos2posemb{pos.shape[-1]}d')(pos, dim)

    def temporal_alignment(self, query_pos, tgt, ref_pts, ref_feat, mem_dict, ref_time=None):
        B = ref_pts.shape[0]
        mem_dict = self.stack_dict_list(mem_dict)
        x = mem_dict['prev_exists'].view(-1)
        # metric coords --> normalized coords
        temp_ref_pts = ((mem_dict['ref_pts'] - self.lidar_range[:self.pos_dim]) /
                        (self.lidar_range[3:3+self.pos_dim] - self.lidar_range[:self.pos_dim]))
        temp_memory = mem_dict['embeddings']

        if not x.all():
            # pad the recent memory ref pts with pseudo points
            ext_inds = torch.randperm(self.topk_ref_pts)[:self.num_propagated]
            ext_ref_pts = ref_pts[:, ext_inds]
            ext_feat = ref_feat[:, ext_inds]
            # pseudo_ref_pts = pseudo_ref_pts + torch.rand_like(pseudo_ref_pts)
            x = x.view(*((-1,) + (1,) * (ext_ref_pts.ndim - 1)))
            temp_ref_pts[:, 0] = temp_ref_pts[:, 0] * x + ext_ref_pts * (1 - x)
            ext_feat = temp_memory[:, 0] * x + ext_feat * (1 - x)
        else:
            ext_feat = temp_memory[:, 0]

        temp_pos = self.query_embedding(self.embed_pos(temp_ref_pts))
        rec_pose = torch.eye(
            4, device=query_pos.device).reshape(1, 1, 4, 4).repeat(
            B, query_pos.size(1), 1, 1)

        # Get ego motion-aware tgt and query_pos for the current frame
        rec_motion = torch.cat(
            [torch.zeros_like(tgt[..., :3]),
             rec_pose[..., :3, :].flatten(-2)], dim=-1)
        rec_motion = PE.nerf_positional_encoding(rec_motion)
        tgt = self.ego_pose_memory(tgt, rec_motion)
        query_pos = self.ego_pose_pe(query_pos, rec_motion)

        # get ego motion-aware reference points embeddings and memory for past frames
        memory_ego_motion = torch.cat(
            [mem_dict['velo'], mem_dict['timestamp'],
             mem_dict['pose'][..., :3, :].flatten(-2)], dim=-1).float()
        memory_ego_motion = PE.nerf_positional_encoding(memory_ego_motion)
        temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
        temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        # get time-aware pos embeddings
        if ref_time is None:
            ref_time = torch.zeros_like(ref_pts[..., :1]) + self.global_ref_time
        query_pos += self.time_embedding(self.embed_pos(ref_time, self.embed_dims))
        temp_pos += self.time_embedding(
            self.embed_pos(mem_dict['timestamp'], self.embed_dims).float())

        tgt = torch.cat([tgt, temp_memory[:, 0]], dim=1)
        query_pos = torch.cat([query_pos, temp_pos[:, 0]], dim=1)
        ref_pts = torch.cat([ref_pts, temp_ref_pts[:, 0]], dim=1)
        # rec_pose = torch.eye(
        #     4, device=query_pos.device).reshape(1, 1, 4, 4).repeat(
        #     B, query_pos.shape[1] + temp_pos[:, 0].shape[1], 1, 1)
        temp_memory = temp_memory[:, 1:].flatten(1, 2)
        temp_pos = temp_pos[:, 1:].flatten(1, 2)

        return tgt, query_pos, ref_pts, temp_memory, temp_pos, ext_feat


class LocalNaiveFusion(BaseModule):
    """This is a naive replacement of LocalTemporalFusion by only selecting the topk points for later spatial fusion"""
    def __init__(self,
                 in_channels,
                 # feature_stride,
                 lidar_range,
                 pos_dim=3,
                 topk_ref_pts=1024,
                 ref_pts_stride=2,
                 transformer_itrs=1,
                 global_ref_time=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        # self.feature_stride = feature_stride
        self.topk_ref_pts = topk_ref_pts
        self.ref_pts_stride = ref_pts_stride
        self.transformer_itrs = transformer_itrs
        self.global_ref_time = global_ref_time

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

    def forward(self, local_roi, bev_feat, mem_dict, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(local_roi, bev_feat, self.ref_pts_stride, self.topk_ref_pts)

        # ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
        #           (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        ref_pos = ref_ctr
        outs_dec = ref_feat[None].repeat(self.transformer_itrs, 1, 1, 1)

        outs = [
            {
                'outs_dec': outs_dec[:, i].permute(1, 0, 2),
                'ref_pts': ref_pos[i],
            } for i in range(len(bev_feat))
        ]

        return {self.scatter_keys[0]: outs}

    def gather_topk(self, rois, bev_feats, stride, topk):
        topk_feat, topk_ctr = [], []
        for roi, bev_feat in zip(rois, bev_feats):
            ctr = bev_feat[f'p{stride}']['ctr']
            feat = bev_feat[f'p{stride}']['feat']
            if 'scr' in roi:
                scores = roi['scr']
            else:
                scores = roi[f'p{stride}']['scr']
            sort_inds = scores.argsort(descending=True)
            if scores.shape[0] < topk:
                n_repeat = topk // len(scores) + 1
                sort_inds = torch.cat([sort_inds] * n_repeat, dim=0)

            topk_inds = sort_inds[:topk]
            topk_ctr.append(ctr[topk_inds])
            topk_feat.append(feat[topk_inds])
        topk_ctr = torch.stack(topk_ctr, dim=0)
        topk_feat = torch.stack(topk_feat, dim=0)
        # pad 2d coordinates to 3d if needed
        if topk_ctr.shape[-1] < self.pos_dim:
            pad_dim = self.pos_dim - topk_ctr.shape[-1]
            topk_ctr = torch.cat([topk_ctr, torch.zeros_like(topk_ctr[..., :pad_dim])], dim=-1)
        return topk_feat, topk_ctr







