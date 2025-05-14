import os
from typing import Mapping, Any

import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import cat_coor_with_idx
from cosense3d.modules.plugin.attn import ScaledDotProductAttention
from cosense3d.modules.utils.common import pad_r
from cosense3d.modules.utils.misc import MLN
import cosense3d.modules.utils.positional_encoding as PE
from cosense3d.agents.utils.transform import filter_range_mask
try:
    from cosense3d.modules.utils.localization_utils import register_points
except:
    pass


class SpatialQueryFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 resolution,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.resolution = resolution
        self.attn = ScaledDotProductAttention(in_channels)

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
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
            out = self.attn(q, kv, kv).squeeze(1)
            fused_feat.append({
                'ref_pts': uniq_coor,
                'outs_dec': out.unsqueeze(1)
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}


class SpatialAlignment(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 voxel_size,
                 stride,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.pix_per_meter = 1 / (voxel_size[0] * stride)
        self.voxel_size = voxel_size[0]
        self.mlp_rotation = nn.Sequential(
            nn.Linear(in_channels + 9, 128),
            nn.LeakyReLU(),
            nn.Linear(128, in_channels),
            nn.LeakyReLU(),
        )
        self.pos_emb = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(in_channels + 64, in_channels),
            nn.LeakyReLU(),
            nn.BatchNorm1d(in_channels),
            # nn.Linear(in_channels, in_channels),
            # nn.LeakyReLU(),
            # nn.BatchNorm1d(in_channels),
        )

    def forward(self, transforms, ego_feats, coop_feats, ego_dets, gts, **kwargs):
        fused_feat = []
        b = 0
        for tf, ego_feat, coop_feat, ego_det, gt in zip(
                transforms, ego_feats, coop_feats, ego_dets, gts):
            coor = [ego_feat['ref_pts']]
            feat = [ego_feat['outs_dec'][:, -1]]
            if len(coop_feat) == 0 or tf is None:
                fused_feat.append({
                        'ref_pts': coor[0][:, :2],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for i, cpfeat in enumerate(coop_feat.values()):
                if tf[i] is None:
                    continue
                coop_pts, coop_feat = self.transform_coop_to_ego(cpfeat, tf[i])
                coor.append(coop_pts)
                feat.append(coop_feat)

            # if kwargs['seq_idx'] ==3 and b== 0:
            #     self.vis_coor(coor, gt)
            #     b += 1
            coor, out = self.nearst_nbr_fusion(coor, feat)
            fused_feat.append({
                'ref_pts': coor,
                'outs_dec': out
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}

    def transform_coop_to_ego(self, cpfeat, T_c2e):
        coop_pts = cpfeat['ref_pts']
        coop_feat = cpfeat['outs_dec'][:, -1]

        coop_pts = (T_c2e @ torch.cat(
            [coop_pts, torch.ones_like(coop_pts[:, :1])
             ], dim=1).T)[:3].T
        coop_pts_ = coop_pts.clone()
        coop_pts_[:, -1] = 0
        mask = filter_range_mask(coop_pts_, self.pc_range)
        coop_pts = coop_pts[mask]
        coop_feat = coop_feat[mask]
        transforms = T_c2e[:3, :3].reshape(1, -1).repeat(len(coop_feat), 1)
        coop_feat = torch.cat((coop_feat, transforms), dim=1)
        coop_feat = self.mlp_rotation(coop_feat)
        return coop_pts, coop_feat

    @torch.no_grad()
    def get_fusion_center_and_nbrs(self, coor_cat):
        coor_ind = coor_cat[:, :2] * self.pix_per_meter
        uniq_coor, inverse_map = torch.unique(coor_ind.floor(), dim=0, return_inverse=True)
        uniq_coor = uniq_coor / self.pix_per_meter

        dist = torch.norm(uniq_coor[:, :2].unsqueeze(1) - coor_cat[:, :2].unsqueeze(0), dim=2)
        _, topk = torch.topk(dist, k=8, largest=False)
        coor_diff = uniq_coor.unsqueeze(1) - coor_cat[topk.view(-1), :2].view(topk.shape + (2, ))
        return uniq_coor, coor_diff, topk

    def nearst_nbr_fusion(self, coor, feat):
        # coor_cat = cat_coor_with_idx(coor)
        coor_cat = torch.cat(coor, dim=0)
        feat_cat = torch.cat(feat, dim=0)
        uniq_coor, coor_diff, topk = self.get_fusion_center_and_nbrs(coor_cat)

        feat = feat_cat[topk.view(-1)]
        feat = torch.cat((feat, self.pos_emb(coor_diff.view(feat.shape[0], -1))), dim=-1)
        feat = self.mlp_fusion(feat).view(topk.shape + (feat_cat.shape[-1],))
        feat = feat.max(dim=1)[0] + feat.mean(dim=1)

        return uniq_coor, feat.unsqueeze(1)

    def attn_fusion(self, coor, feat):
        q, kv, uniq_coor = self.prepare_qkv(coor, feat)
        out = self.attn(q, kv, kv).squeeze(1)
        return uniq_coor, out.unsqueeze(1)

    def prepare_qkv(self, coor, feat):
        coor_cat = cat_coor_with_idx(coor)
        feat_cat = torch.cat(feat, dim=0)

        coor_ind = coor_cat[:, 1:3]
        coor_ind = coor_ind * self.pix_per_meter
        # coor_ind[:, :2] = ((coor_ind[:, :2] - self.voxel_size / 2) * self.pix_per_meter)
        uniq_coor, inverse_map = torch.unique(coor_ind.floor(), dim=0,
                                               return_inverse=True)

        # uniq_coor[:, :2] = uniq_coor[:, :2] / self.pix_per_meter + self.voxel_size / 2
        pos_diff = coor_ind - uniq_coor[inverse_map]
        pos_diff_emb = self.pos_diff_emb(PE.pos2posemb2d(pos_diff, 64))
        feat_cat = feat_cat + pos_diff_emb
        uniq_coor = uniq_coor / self.pix_per_meter

        feats_pad = []
        for i, c in enumerate(coor):
            feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
            feat_pad[inverse_map[coor_cat[:, 0] == i]] = feat[i]
            feats_pad.append(feat_pad)
        q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
        kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
        return q, kv, uniq_coor

    def vis_coor(self, coor, gt):
        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        ax = draw_points_boxes_plt(
            pc_range=self.pc_range.tolist(),
            boxes_gt=gt,
            return_ax=True
        )
        for p in coor:
            p = p.detach().cpu().numpy()
            ax.plot(p[:, 0], p[:, 1], '.', markersize=1)
        plt.savefig(f"{os.environ.get('HOME')}/Downloads/tmp2.jpg")
        plt.close()


class SpatialAttentiveAlignment(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 voxel_size,
                 stride,
                 emb_dim=256,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.pix_per_meter = 1 / (voxel_size[0] * stride)
        self.voxel_size = voxel_size[0]
        self.emb_dim = emb_dim
        self.mlp_rotation = nn.Sequential(
            nn.Linear(in_channels + 9, 128),
            nn.LeakyReLU(),
            nn.Linear(128, in_channels),
            nn.LeakyReLU(),
        )
        self.pos_emb = nn.Sequential(
            nn.Linear(2, emb_dim),
            nn.LeakyReLU(),
        )
        # self.feat_emb = nn.Sequential(
        #     nn.Linear(in_channels, emb_dim),
        #     nn.LeakyReLU(),
        # )
        self.mha = nn.MultiheadAttention(emb_dim, 8, batch_first=True)
        # self.out_layer = nn.Sequential(
        #     nn.Linear(emb_dim, in_channels),
        #     nn.LeakyReLU(),
        # )

    def forward(self, transforms, ego_feats, coop_feats, ego_dets, gts, **kwargs):
        fused_feat = []
        b = 0
        for tf, ego_feat, coop_feat, ego_det, gt in zip(
                transforms, ego_feats, coop_feats, ego_dets, gts):
            coor = [ego_feat['ref_pts']]
            feat = [ego_feat['outs_dec'][:, -1]]
            if len(coop_feat) == 0 or tf is None:
                fused_feat.append({
                        'ref_pts': coor[0][:, :2],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for i, cpfeat in enumerate(coop_feat.values()):
                if tf[i] is None:
                    continue
                coop_pts, coop_feat = self.transform_coop_to_ego(cpfeat, tf[i])
                coor.append(coop_pts)
                feat.append(coop_feat)

            # if kwargs['seq_idx'] ==3 and b== 0:
            #     self.vis_coor(coor, gt)
            #     b += 1
            coor, out = self.nearst_nbr_fusion(coor, feat)
            fused_feat.append({
                'ref_pts': coor,
                'outs_dec': out
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}

    def transform_coop_to_ego(self, cpfeat, T_c2e):
        coop_pts = cpfeat['ref_pts']
        coop_feat = cpfeat['outs_dec'][:, -1]

        coop_pts = (T_c2e @ torch.cat(
            [coop_pts, torch.ones_like(coop_pts[:, :1])
             ], dim=1).T)[:3].T
        coop_pts_ = coop_pts.clone()
        coop_pts_[:, -1] = 0
        mask = filter_range_mask(coop_pts_, self.pc_range)
        coop_pts = coop_pts[mask]
        coop_feat = coop_feat[mask]
        transforms = T_c2e[:3, :3].reshape(1, -1).repeat(len(coop_feat), 1)
        coop_feat = torch.cat((coop_feat, transforms), dim=1)
        coop_feat = self.mlp_rotation(coop_feat)
        return coop_pts, coop_feat

    @torch.no_grad()
    def get_fusion_center_and_nbrs(self, coor_cat):
        coor_ind = coor_cat[:, :2] * self.pix_per_meter
        uniq_coor, inverse_map = torch.unique(coor_ind.floor(), dim=0, return_inverse=True)
        uniq_coor = uniq_coor / self.pix_per_meter

        dist = torch.norm(uniq_coor[:, :2].unsqueeze(1) - coor_cat[:, :2].unsqueeze(0), dim=2)
        _, topk = torch.topk(dist, k=8, largest=False)
        coor_diff = uniq_coor.unsqueeze(1) - coor_cat[topk.view(-1), :2].view(topk.shape + (2, ))
        return uniq_coor, coor_diff, topk

    def nearst_nbr_fusion(self, coor, feat):
        # coor_cat = cat_coor_with_idx(coor)
        coor_cat = torch.cat(coor, dim=0)
        feat_cat = torch.cat(feat, dim=0)
        uniq_coor, coor_diff, topk = self.get_fusion_center_and_nbrs(coor_cat)

        # feat_cat = self.feat_emb(feat_cat)
        feat = feat_cat[topk.view(-1)].view(topk.shape + (self.emb_dim,))
        pos_emb = self.pos_emb(coor_diff.view(-1, 2)).view(topk.shape + (self.emb_dim,))
        kv = feat + pos_emb
        tgt = torch.zeros_like(kv[:, :1])
        attn_out, attn_weight = self.mha(tgt, kv, kv)

        # attn_out = self.out_layer(attn_out.squeeze(1)).unsqueeze(1)
        return uniq_coor, attn_out

    def attn_fusion(self, coor, feat):
        q, kv, uniq_coor = self.prepare_qkv(coor, feat)
        out = self.attn(q, kv, kv).squeeze(1)
        return uniq_coor, out.unsqueeze(1)

    def prepare_qkv(self, coor, feat):
        coor_cat = cat_coor_with_idx(coor)
        feat_cat = torch.cat(feat, dim=0)

        coor_ind = coor_cat[:, 1:3]
        coor_ind = coor_ind * self.pix_per_meter
        # coor_ind[:, :2] = ((coor_ind[:, :2] - self.voxel_size / 2) * self.pix_per_meter)
        uniq_coor, inverse_map = torch.unique(coor_ind.floor(), dim=0,
                                               return_inverse=True)

        # uniq_coor[:, :2] = uniq_coor[:, :2] / self.pix_per_meter + self.voxel_size / 2
        pos_diff = coor_ind - uniq_coor[inverse_map]
        pos_diff_emb = self.pos_diff_emb(PE.pos2posemb2d(pos_diff, 64))
        feat_cat = feat_cat + pos_diff_emb
        uniq_coor = uniq_coor / self.pix_per_meter

        feats_pad = []
        for i, c in enumerate(coor):
            feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
            feat_pad[inverse_map[coor_cat[:, 0] == i]] = feat[i]
            feats_pad.append(feat_pad)
        q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
        kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
        return q, kv, uniq_coor

    def vis_coor(self, coor, gt):
        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        ax = draw_points_boxes_plt(
            pc_range=self.pc_range.tolist(),
            boxes_gt=gt,
            return_ax=True
        )
        for p in coor:
            p = p.detach().cpu().numpy()
            ax.plot(p[:, 0], p[:, 1], '.', markersize=1)
        plt.savefig(f"{os.environ.get('HOME')}/Downloads/tmp2.jpg")
        plt.close()


class SpatialQueryAlignFusionRL(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 resolution,
                 num_pose_feat=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.resolution = resolution
        self.emb_dim = in_channels
        self.attn = ScaledDotProductAttention(in_channels)
        self.pose_pe = MLN(4 * 12, f_dim=self.emb_dim)
        self.num_pose_feat = num_pose_feat
        self.position_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(self, det_local, roadline, roadline_preds, ego_queries,
                ego_pose_corrected, ego_poses, ego_poses_aug,
                cpms, **kwargs):
        fused_feat = []
        for i, cpm in enumerate(cpms):
            det = det_local[i]
            ego_rl, ego_rl_pred, ego_query = roadline[i], roadline_preds[i], ego_queries[i]
            ego_pose_corr, ego_pose, pose_aug2g = ego_pose_corrected[i], ego_poses[i], ego_poses_aug[i]
            # augment-frame to ego-aligned-world frame
            Taug2eaw = ego_pose_corr @ ego_pose.inverse() @ pose_aug2g
            ego_bctr = det['preds']['box'][:, :2]
            ego_coor = ego_query['ref_pts']
            ego_coor_emb = self.query_embedding(PE.pos2posemb2d(ego_coor[:, :2], self.num_pose_feat))
            ego_feat = ego_query['outs_dec'][-1] + ego_coor_emb
            ego_coor = ego_coor * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            coor = [ego_coor] # in augment-frame
            feat = [ego_feat]
            if len(cpm) == 0:
                fused_feat.append({
                        'ref_pts': coor[0],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for cpfeat in cpm.values():
                if len(cpfeat['box_ctrs']) == 0:
                    continue
                # transformation matrix coop-aligned-world frame to ego-aligned-world frame
                if self.training:
                    # during training, ground-truth poses are used, caw-frame==eaw-frame
                    Tcaw2aug = Taug2eaw.inverse()
                else:
                    Tcaw2eaw = self.align_coordinates(ego_bctr, ego_rl, ego_rl_pred, Taug2eaw, cpfeat)
                    Tcaw2aug = Taug2eaw.inverse() @ Tcaw2eaw
                T = Tcaw2aug @ cpfeat['Taug2caw']

                # encode the transformation matrix that transforms feature points
                # from erroneous ego-frame to the corrected ego-frame
                ref_pts = (T @ pad_r(cpfeat['ref_pts'], 1.0).T)[:3].T
                ref_pts_norm = (ref_pts - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
                rot_emb = PE.nerf_positional_encoding(T[:2, :2].flatten(-2)).repeat(len(ref_pts), 1)
                pos_emb = self.position_embedding(PE.pos2posemb2d(ref_pts_norm[:, :2], self.num_pose_feat))
                transform_emb = self.pose_pe(pos_emb, rot_emb)
                coor.append(ref_pts)
                feat.append(cpfeat['feat'][-1] + transform_emb)

                # inplace transformation for coop point cloud: only for visualization in GLViewer
                cpfeat['points'][:, :3] = (T @ pad_r(cpfeat['points'][:, :3], 1.0).T)[:3].T

            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            # coor_int = coor_cat[:, 1:] * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            coor_int = (coor_cat[:, 1:] * (1 / self.resolution)).int()
            uniq_coor, reverse_inds = torch.unique(coor_int, dim=0, return_inverse=True)
            uniq_coor = (uniq_coor * self.resolution - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
            out = self.attn(q, kv, kv).squeeze(1)
            fused_feat.append({
                'ref_pts': uniq_coor,
                'outs_dec': out.unsqueeze(1)
            })

            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            # ax = draw_points_boxes_plt(pc_range=self.pc_range.tolist(), return_ax=True)
            # for pts in coor:
            #     pts = pts.detach().cpu().numpy()
            #     ax.plot(pts[:, 0], pts[:, 1], '.', markersize=1)
            # plt.savefig("/home/yuan/Downloads/tmp.png")
            # plt.close()
        return self.format_output(fused_feat)

    def format_output(self, output, **kwargs):
        return {self.scatter_keys[0]: output}

    def align_coordinates(self, ego_bctr, ego_rl, ego_rl_pred, ego_pose, cpfeat):
        coop_bctr = cpfeat['box_ctrs']
        coop_rl = cpfeat['roadline']

        # transform ego points from aug-frame to ego-aligned world-frame
        ego_bctr = (ego_pose @ pad_r(pad_r(ego_bctr, 0.0), 1.0).T).T
        ego_rl_pred = (ego_pose @ pad_r(pad_r(ego_rl_pred, 0.0), 1.0).T).T
        coop_pts = pad_r(torch.cat([coop_rl, coop_bctr], dim=0))
        ego_pts = torch.cat([pad_r(ego_rl[:, :3]), ego_bctr[:, :3]], dim=0)

        transform, coop_pts_tf = register_points(coop_pts, ego_pts, thr=0.8)

        # import matplotlib.pyplot as plt
        # ego_bctr_vis = ego_bctr.detach().cpu().numpy()
        # ego_rl_pred_vis = ego_rl_pred.detach().cpu().numpy()
        # ego_rl_vis = ego_rl.detach().cpu().numpy()
        # coop_bctr_vis = coop_bctr.detach().cpu().numpy()
        # coop_rl_vis = coop_rl.detach().cpu().numpy()
        #
        # plt.plot(ego_rl_vis[:, 0], ego_rl_vis[:, 1], 'g.', markersize=1)
        # plt.plot(ego_rl_pred_vis[:, 0], ego_rl_pred_vis[:, 1], 'y.', markersize=1)
        # plt.plot(ego_bctr_vis[:, 0], ego_bctr_vis[:, 1], 'yo', markersize=5, markerfacecolor='none')
        # plt.plot(coop_rl_vis[:, 0], coop_rl_vis[:, 1], 'r.', markersize=1)
        # plt.plot(coop_bctr_vis[:, 0], coop_bctr_vis[:, 1], 'ro', markersize=5, markerfacecolor='none', alpha=0.5)
        # # plt.plot(coop_pts_tf[:, 0], coop_pts_tf[:, 1], 'b.', markersize=1)
        # plt.savefig("/home/yys/Downloads/tmp.png")
        # plt.close()

        return torch.from_numpy(transform).float().to(ego_pose.device)













