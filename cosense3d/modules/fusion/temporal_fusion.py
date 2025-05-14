from typing import Mapping, Any

import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.misc import SELayer_Linear, MLN, MLN2
import cosense3d.modules.utils.positional_encoding as PE


class TemporalLidarFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
                 lidar_range,
                 pos_dim=3,
                 num_pose_feat=64,
                 topk=2048,
                 num_propagated=256,
                 memory_len=1024,
                 num_query=644,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.feature_stride = feature_stride
        self.topk = topk
        self.num_query = num_query
        self.num_propagated = num_propagated
        self.memory_len = memory_len

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.position_embeding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        self.reference_points = nn.Embedding(self.num_query, self.pos_dim)
        self.pseudo_reference_points = nn.Embedding(self.num_propagated, self.pos_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        pose_nerf_dim = (3 + 3 * 4) * 12
        self.ego_pose_pe = MLN(pose_nerf_dim, f_dim=self.embed_dims)
        self.ego_pose_memory = MLN(pose_nerf_dim, f_dim=self.embed_dims)

    def init_weights(self):
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
        self.pseudo_reference_points.weight.requires_grad = False
        self.transformer.init_weights()

    def forward(self, rois, bev_feat, mem_dict, **kwargs):
        feat, ctr = self.gather_topk(rois, bev_feat)

        pos = ((ctr - self.lidar_range[:self.pos_dim]) /
               (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        pos_emb = self.position_embeding(self.embed_pos(pos))
        memory = self.memory_embed(feat)
        pos_emb = self.featurized_pe(pos_emb, memory)

        reference_points = self.reference_points.weight.unsqueeze(0).repeat(memory.shape[0], 1, 1)
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos = \
            self.temporal_alignment(query_pos, tgt, reference_points, mem_dict)
        mask_dict = [None, None]
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_emb,
                                       mask_dict, temp_memory, temp_pos)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(rois))
        ]

        return {self.scatter_keys[0]: outs}

    def gather_topk(self, rois, bev_feats):
        topk_feat, topk_ctr = [], []
        for roi, bev_feat in zip(rois, bev_feats):
            ctr = bev_feat[f'p{self.feature_stride}']['ctr']
            feat = bev_feat[f'p{self.feature_stride}']['feat']
            scores = roi['scr']
            if scores.shape[0] < self.topk:
                raise NotImplementedError
            else:
                topk_inds = torch.topk(scores, k=self.topk).indices
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

    def temporal_alignment(self, query_pos, tgt, ref_pts, mem_dict):
        B = ref_pts.shape[0]
        mem_dict = self.stack_dict_list(mem_dict)
        x = mem_dict['prev_exists'].view(-1)
        # metric coords --> normalized coords
        temp_ref_pts = ((mem_dict['ref_pts'] - self.lidar_range[:self.pos_dim]) /
                        (self.lidar_range[3:3+self.pos_dim] - self.lidar_range[:self.pos_dim]))
        if not x.all():
            # pad the recent memory ref pts with pseudo points
            pseudo_ref_pts = self.pseudo_reference_points.weight.unsqueeze(0).repeat(B, 1, 1)
            x = x.view(*((-1,) + (1,) * (pseudo_ref_pts.ndim - 1)))
            temp_ref_pts[:, 0] = temp_ref_pts[:, 0] * x + pseudo_ref_pts * (1 - x)

        temp_pos = self.query_embedding(self.embed_pos(temp_ref_pts))
        temp_memory = mem_dict['embeddings']
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
        query_pos += self.time_embedding(
            self.embed_pos(torch.zeros_like(ref_pts[..., :1]), self.embed_dims))
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

        return tgt, query_pos, ref_pts, temp_memory, temp_pos


class TemporalFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
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
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.feature_stride = feature_stride
        self.topk_ref_pts = topk_ref_pts
        self.topk_feat = topk_feat
        self.ref_pts_stride = ref_pts_stride
        self.num_propagated = num_propagated
        self.memory_len = memory_len
        self.transformer_itrs = transformer_itrs
        self.global_ref_time = global_ref_time

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.position_embeding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims * 4),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

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

    def forward(self, rois, bev_feat, mem_dict, time_scale=None, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(rois, bev_feat, self.ref_pts_stride, self.topk_ref_pts)
        mem_feat, mem_ctr = self.gather_topk(rois, bev_feat, self.feature_stride, self.topk_feat)

        ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos = ((mem_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos_emb = self.position_embeding(self.embed_pos(mem_pos))
        memory = self.memory_embed(mem_feat)
        pos_emb = self.featurized_pe(mem_pos_emb, memory)

        if time_scale is not None:
            ref_time = torch.rad2deg(torch.arctan2(ref_ctr[..., 1:2], ref_ctr[..., 0:1])) + 180
            ref_time = torch.stack([ts[inds.long()] for inds, ts in zip(ref_time, time_scale)], dim=0)
        else:
            ref_time = None
        reference_points = ref_pos.clone()
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos, ext_feat = \
            self.temporal_alignment(query_pos, tgt, reference_points,
                                    ref_feat, mem_dict, ref_time)
        mask_dict = [None, None]
        global_feat = []

        for _ in range(self.transformer_itrs):
            tgt = self.transformer(memory, tgt, query_pos, pos_emb,
                                   mask_dict, temp_memory, temp_pos)[0][-1]
            global_feat.append(tgt)
        global_feat = torch.stack(global_feat, dim=0)
        local_feat = torch.cat([ref_feat, ext_feat], dim=1)
        local_feat = local_feat[None].repeat(self.transformer_itrs, 1, 1, 1)
        outs_dec = local_feat + global_feat

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(rois))
        ]

        return {self.scatter_keys[0]: outs}

    def gather_topk(self, rois, bev_feats, stride, topk):
        topk_feat, topk_ctr = [], []
        for roi, bev_feat in zip(rois, bev_feats):
            ctr = bev_feat[f'p{stride}']['ctr']
            feat = bev_feat[f'p{stride}']['feat']
            scores = roi[f'p{stride}']['conf'][:,
                     roi[f'p{stride}']['reg'].shape[-1] - 1:].sum(dim=-1)
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


class LocalTemporalFusion(BaseModule):
    """Modified from TemporalFusion to standardize input and output keys"""
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
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
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.feature_stride = feature_stride
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
        if self.norm_fusion:
            self.local_global_fusion = nn.Sequential(
                nn.Linear(self.embed_dims * 2, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.position_embeding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims * 4),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

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

    def forward(self, local_roi, global_roi, bev_feat, mem_dict, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(local_roi, bev_feat, self.ref_pts_stride, self.topk_ref_pts)
        mem_feat, mem_ctr = self.gather_topk(global_roi, bev_feat, self.feature_stride, self.topk_feat)

        ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos = ((mem_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos_emb = self.position_embeding(self.embed_pos(mem_pos))
        memory = self.memory_embed(mem_feat)
        pos_emb = self.featurized_pe(mem_pos_emb, memory)

        ref_time = None
        reference_points = ref_pos.clone()
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos, ext_feat = \
            self.temporal_alignment(query_pos, tgt, reference_points,
                                    ref_feat, mem_dict, ref_time)
        mask_dict = [None, None]
        global_feat = []

        for _ in range(self.transformer_itrs):
            tgt = self.transformer(memory, tgt, query_pos, pos_emb,
                                   mask_dict, temp_memory, temp_pos)[0][-1]
            global_feat.append(tgt)
        global_feat = torch.stack(global_feat, dim=0)
        local_feat = torch.cat([ref_feat, ext_feat], dim=1)
        local_feat = local_feat[None].repeat(self.transformer_itrs, 1, 1, 1)
        if self.norm_fusion:
            outs_dec = self.local_global_fusion(torch.cat([local_feat, global_feat], dim=-1))
        else:
            # simple addition will lead to large values in long sequences
            outs_dec = local_feat + global_feat

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
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


class LocalTemporalFusionV1(LocalTemporalFusion):
    def forward(self, rois, bev_feat, mem_dict, **kwargs):
        return super().forward(rois, rois, bev_feat, mem_dict, **kwargs)


class LocalTemporalFusionV2(LocalTemporalFusion):
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
        mask_dict = None
        global_feat = []

        for _ in range(self.transformer_itrs):
            tgt = self.transformer(None, tgt, query_pos, None,
                                   mask_dict, temp_memory, temp_pos)[0][-1]
            global_feat.append(tgt)
        global_feat = torch.stack(global_feat, dim=0)
        local_feat = torch.cat([ref_feat, ext_feat], dim=1)
        local_feat = local_feat[None].repeat(self.transformer_itrs, 1, 1, 1)
        outs_dec = local_feat + global_feat

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(bev_feat))
        ]

        return {self.scatter_keys[0]: outs}


class LocalTemporalFusionV3(BaseModule):
    """TemporalFusion with feature flow"""
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
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
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = num_pose_feat
        self.pos_dim = pos_dim
        self.in_channels = in_channels
        self.feature_stride = feature_stride
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
        if self.norm_fusion:
            self.local_global_fusion = nn.Sequential(
                nn.Linear(self.embed_dims * 2, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.LayerNorm(self.embed_dims),
            )

        self.position_embeding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims * 4),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims * 4),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.memory_embed = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * self.pos_dim, self.embed_dims),
            nn.ReLU(),
            nn.LayerNorm(self.embed_dims),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

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

    def forward(self, local_roi, global_roi, bev_feat, mem_dict, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(local_roi, bev_feat, self.ref_pts_stride, self.topk_ref_pts)
        mem_feat, mem_ctr = self.gather_topk(global_roi, bev_feat, self.feature_stride, self.topk_feat)

        ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos = ((mem_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        mem_pos_emb = self.position_embeding(self.embed_pos(mem_pos))
        memory = self.memory_embed(mem_feat)
        pos_emb = self.featurized_pe(mem_pos_emb, memory)

        ref_time = None
        reference_points = ref_pos.clone()
        query_pos = self.query_embedding(self.embed_pos(reference_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temp_memory, temp_pos, ext_feat = \
            self.temporal_alignment(query_pos, tgt, reference_points,
                                    ref_feat, mem_dict, ref_time)
        mask_dict = [None, None]
        global_feat = []

        for _ in range(self.transformer_itrs):
            tgt = self.transformer(memory, tgt, query_pos, pos_emb,
                                   mask_dict, temp_memory, temp_pos)[0][-1]
            global_feat.append(tgt)
        global_feat = torch.stack(global_feat, dim=0)
        local_feat = torch.cat([ref_feat, ext_feat], dim=1)
        local_feat = local_feat[None].repeat(self.transformer_itrs, 1, 1, 1)
        if self.norm_fusion:
            outs_dec = self.local_global_fusion(torch.cat([local_feat, global_feat], dim=-1))
        else:
            # simple addition will lead to large values in long sequences
            outs_dec = local_feat + global_feat

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
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
                 feature_stride,
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
        self.feature_stride = feature_stride
        self.topk_ref_pts = topk_ref_pts
        self.ref_pts_stride = ref_pts_stride
        self.transformer_itrs = transformer_itrs
        self.global_ref_time = global_ref_time

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)

    def forward(self, local_roi, global_roi, bev_feat, mem_dict, **kwargs):
        ref_feat, ref_ctr = self.gather_topk(local_roi, bev_feat, self.ref_pts_stride, self.topk_ref_pts)

        ref_pos = ((ref_ctr - self.lidar_range[:self.pos_dim]) /
                  (self.lidar_range[3:self.pos_dim + 3] - self.lidar_range[:self.pos_dim]))
        outs_dec = ref_feat[None].repeat(self.transformer_itrs, 1, 1, 1)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
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





