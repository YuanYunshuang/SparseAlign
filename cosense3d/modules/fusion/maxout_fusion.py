import torch
import torch.nn as nn

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.common import cat_coor_with_idx


class BEVMaxoutFusion(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ego_feats, coop_feats, **kwargs):
        out_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            feat = [ego_feat]
            for cpfeat in coop_feat.values():
                if 'bev_feat' not in cpfeat:
                    continue
                feat.append(cpfeat['bev_feat'])
            feat = torch.stack(feat, dim=0).max(dim=0).values
            out_feat.append(feat)

        return {self.scatter_keys[0]: out_feat}


class SparseBEVMaxoutFusion(BaseModule):
    def __init__(self,
                 pc_range,
                 resolution,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.resolution = resolution

    def forward(self, ego_feats, coop_feats, **kwargs):
        fused_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            coor = [ego_feat['ref_pts']]
            feat = [ego_feat['outs_dec'][-1]]
            if len(coop_feat) == 0:
                fused_feat.append({
                        'ref_pts': coor[0],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for cpfeat in coop_feat.values():
                coor.append(cpfeat[self.gather_keys[0]]['ref_pts'])
                feat.append(cpfeat[self.gather_keys[0]]['outs_dec'][-1])
            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            # coor_int = coor_cat[:, 1:] * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            # coor_int = (coor_int * (1 / self.resolution)).int()
            uniq_coor, reverse_inds = torch.unique(coor_cat[:, 1:], dim=0,
                                                   return_inverse=True)

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_full((len(uniq_coor), feat_cat.shape[-1]), -torch.inf)
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            out = torch.stack(feats_pad, dim=0).max(dim=0).values
            fused_feat.append({
                'ref_pts': uniq_coor,
                'outs_dec': out.unsqueeze(1)
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}