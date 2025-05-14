import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.misc import SELayer_Linear, MLN
from cosense3d.modules.utils.positional_encoding import pos2posemb2d


class LidarPETRHead(BaseModule):
    def __init__(self,
                 in_channels,
                 transformer,
                 feature_stride,
                 lidar_range,
                 topk=2048,
                 memory_len=256,
                 num_query=644,
                 **kwargs):
        super().__init__(**kwargs)
        self.transformer = plugin.build_plugin_module(transformer)
        self.embed_dims = self.transformer.embed_dims
        self.num_pose_feat = 64
        self.pos_dim = 2
        self.in_channels = in_channels
        self.feature_stride = feature_stride
        self.topk = topk
        self.num_query = num_query

        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)
        self.reference_points = nn.Embedding(self.num_query, self.pos_dim)

        self._init_layers()

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

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, rois, bev_feat, memory, **kwargs):
        feat, ctr = self.gather_topk(rois, bev_feat)

        pos = ((ctr - self.lidar_range[:2]) /
               (self.lidar_range[3:5] - self.lidar_range[:2]))
        pos_emb = self.position_embeding(pos2posemb2d(pos, self.num_pose_feat))
        memory = self.memory_embed(feat)
        pos_emb = self.featurized_pe(pos_emb, memory)

        reference_points = (self.reference_points.weight).unsqueeze(0).repeat(memory.shape[0], 1, 1)
        query_pos = self.query_embedding(pos2posemb2d(reference_points, self.num_pose_feat))
        tgt = torch.zeros_like(query_pos)
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_emb)

        outs = [
            {
                'outs_dec': outs_dec[:, i],
                'ref_pts': reference_points[i],
            } for i in range(len(rois))
        ]

        return {self.scatter_keys[0]: outs}

    def format_input(self, input):
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
        return topk_feat, topk_ctr

