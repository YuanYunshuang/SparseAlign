import warnings
from typing import Dict

import torch

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.plugin.attn import ScaledDotProductAttention
from cosense3d.modules.utils.me_utils import update_me_essentials
from cosense3d.modules.utils.common import cat_coor_with_idx


class SparseAttentionFusion(BaseModule):
    def __init__(self, stride, in_channels, **kwargs):
        super(SparseAttentionFusion, self).__init__(**kwargs)
        if isinstance(stride, int):
            self.stride = [stride]
        else:
            self.stride = stride
        self.attn = ScaledDotProductAttention(in_channels)

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        fused_feat = []
        fuse_key = self.gather_keys[0]
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            batch_feat = {}
            for stride in self.stride:
                coor, feat, ctr = self.fuse_feature_at_stride(ego_feat, coop_feat, stride, fuse_key)
                batch_feat[f'p{stride}'] = {'coor': coor, 'feat': feat, 'ctr': ctr}
            fused_feat.append(batch_feat)
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}

    def fuse_feature_at_stride(self, ego_feat, coop_feat, stride, fuse_key):
        coor = [ego_feat[f'p{stride}']['coor']]
        feat = [ego_feat[f'p{stride}']['feat']]
        ctr = [ego_feat[f'p{stride}']['ctr']]
        if len(coop_feat) == 0:
            return coor[0], feat[0], ctr[0]
        else:
            # fuse coop to ego
            for cpfeat in coop_feat.values():
                if fuse_key not in cpfeat:
                    continue
                cpm = cpfeat[fuse_key][f'p{stride}']
                coor.append(cpm['coor'])
                feat.append(cpm['feat'])
                ctr.append(cpm['ctr'])

            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            ctr_cat = torch.cat(ctr, dim=0)
            uniq_coor, reverse_inds = torch.unique(coor_cat[:, 1:], dim=0,
                                                   return_inverse=True)
            uniq_ctr = ctr_cat[reverse_inds.unique()]

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad[1:], dim=1)  # num_pts, num_coop_cav, d
            feat_out = self.attn(q, kv, kv).squeeze(1)
            return uniq_coor, feat_out, uniq_ctr


class DenseAttentionFusion(BaseModule):
    def __init__(self, feature_dim, neck=None, **kwargs):
        super(DenseAttentionFusion, self).__init__(**kwargs)
        self.attn = ScaledDotProductAttention(feature_dim)
        if neck is not None:
            self.neck = plugin.build_plugin_module(neck)

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        out = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            feat = [ego_feat]
            for cpfeat in coop_feat.values():
                if 'bev_feat' not in cpfeat:
                    continue
                feat.append(cpfeat['bev_feat'])
            xx = torch.stack(feat, dim=0)
            N, C, H, W = xx.shape
            xx = xx.view(N, C, -1).permute(2, 0, 1)
            h = self.attn(xx, xx, xx)
            h = h.permute(1, 2, 0).view(N, C, H, W)[0, ...]
            out.append(h)
        out = torch.stack(out)
        if hasattr(self, 'neck'):
            out = self.neck(out)
        return {self.scatter_keys[0]: out}

