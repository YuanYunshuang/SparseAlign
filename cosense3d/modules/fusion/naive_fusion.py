import warnings
from typing import Dict

import torch

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import update_me_essentials


class NaiveFusion(BaseModule):
    def __init__(self, stride, **kwargs):
        super(NaiveFusion, self).__init__(**kwargs)
        if isinstance(stride, int):
            self.stride = [stride]
        else:
            self.stride = stride

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        fused_feat = []
        fuse_key = self.gather_keys[0]

        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            batch_feat = {}
            for stride in self.stride:
                coor, feat, ctr = self.fuse_feature_at_stride(
                    ego_feat, coop_feat, stride, fuse_key
                )
                batch_feat[f'p{stride}'] = {
                        'coor': coor,
                        'feat': feat,
                        'ctr': ctr,
                    }
            fused_feat.append(batch_feat)
        return self.format_output(fused_feat)

    def fuse_feature_at_stride(self, ego_feat, coop_feat, stride, fuse_key):
        coor = [ego_feat[f'p{stride}']['coor']]
        feat = [ego_feat[f'p{stride}']['feat']]
        ctr = [ego_feat[f'p{stride}']['ctr']]
        # fuse coop to ego
        for cpfeat in coop_feat.values():
            if fuse_key not in cpfeat:
                continue
            cpm = cpfeat[fuse_key][f'p{stride}']
            coor.append(cpm['coor'])
            feat.append(cpm['feat'])
            ctr.append(cpm['ctr'])
        coor = torch.cat(coor, dim=0)
        feat = torch.cat(feat, dim=0)
        ctr = torch.cat(ctr, dim=0)
        return coor, feat, ctr


    def format_output(self, output):
        return {self.scatter_keys[0]: output}





