import copy
import torch

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.common import pad_r
from cosense3d.modules.utils.localization_utils import register_points
from cosense3d.utils.pclib import tf2pose, project_points_by_matrix_torch


class SpatialAlignment(BaseModule):
    def __init__(self, **kwargs):
        super(SpatialAlignment, self).__init__(**kwargs)

    def forward(self, dets_local, roadline_pred, feats, **kwargs):
        for det, rl, rl_ref, feat in zip(dets_local, roadline_pred, feats):
            det_ctr = det['preds']['box'][:, :3]
            rl_pts = self.roadline_map_to_points(rl)

            import matplotlib.pyplot as plt
            pts0 = det_ctr.detach().cpu().numpy()
            pts1 = rl_pts.detach().cpu().numpy()
            plt.plot(pts0[:, 0], pts0[:, 1], 'g.')
            plt.plot(pts1[:, 0], pts1[:, 1], 'r.')
            plt.show()
            plt.close()

    def roadline_map_to_points(self, roadline_map):
        scr = roadline_map['cls'].sigmoid().squeeze()
        pos = scr > 0.5
        return roadline_map['ctr'][pos]


class MapRegistration(BaseModule):
    """
    Register local detected roadline points into global roadline map.
    """
    def __init__(self, **kwargs):
        super(MapRegistration, self).__init__(**kwargs)
        self.seq_len = 4

    def forward(self, roadline, road_line_ref, poses_err, poses_gt=None, **kwargs):
        """
        Register local detected roadline points into global roadline map.

        :param roadline: dict, coor (Nx2, voxel indices), ctr (Nx2, voxel center coordinates in meter),
            cls (NxC, classification logits)
        :param road_line_ref: Nx2, ground-truth BEV roadline points in global world coordinates.
        :param poses_err: 4x4, LiDARs' erroneous poses in global world coordinates.
        :param poses_gt: 4x4, LiDARs' ground-truth poses in global world coordinates.
        :param kwargs:
        :return:
            - poses_corrected: 4x4, corrected LiDAR poses.
            - roadline_preds: Nx2, BEV roadline points in local LiDAR coordinates.
        """
        poses_corrected = []
        roadline_preds = []
        for i, rl in enumerate(roadline):
            rl_ref = road_line_ref[i]  # in world-frame
            rl_pts = self.roadline_map_to_points(rl)
            roadline_preds.append(rl_pts)

            if self.training:
                poses_corrected.append(poses_gt[i])
            else:
                pose = poses_err[i]
                rl_pts = pad_r(copy.deepcopy(rl_pts), 0.0)
                rl_ref = pad_r(rl_ref)
                pose_corr, rl_pts_tf = register_points(rl_pts, rl_ref, pose)
                pose_corr = torch.from_numpy(pose_corr).float().to(pose.device)
                poses_corrected.append(pose_corr)

            # import matplotlib.pyplot as plt
            # pts0 = rl_ref.detach().cpu().numpy()
            # pts1 = project_points_by_matrix_torch(rl_pts, pose).detach().cpu().numpy()
            # plt.plot(pts0[:, 0], pts0[:, 1], 'g.', markersize=1)
            # plt.plot(pts1[:, 0], pts1[:, 1], 'r.', markersize=1)
            # plt.plot(rl_pts_tf[:, 0], rl_pts_tf[:, 1], 'b.', markersize=1)
            # plt.show()
            # plt.close()
        return {self.scatter_keys[0]: poses_corrected,
                self.scatter_keys[1]: roadline_preds}

    def roadline_map_to_points(self, roadline_map):
        """Parse roadline detection results to 2d BEV points."""
        scr = roadline_map['cls'].sigmoid().squeeze()
        pos = scr > 0.5
        return roadline_map['ctr'][pos]


