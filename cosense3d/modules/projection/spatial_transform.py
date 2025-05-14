import torch.nn as nn
import torch
from einops import rearrange
from cosense3d.modules import BaseModule
from cosense3d.modules.utils import cobevt_utils as utils


class STTF(BaseModule):
    def __init__(self,
                 resolution,
                 downsample_rate,
                 use_roi_mask=True,
                 **kwargs):
        super(STTF, self).__init__(**kwargs)
        self.discrete_ratio = resolution
        self.downsample_rate = downsample_rate
        self.use_roi_mask = use_roi_mask

    def forward(self, bev_feat, requests, coop_poses, **kwargs):
        """
        Transform the bev features to ego space.
        """
        x = self.stack_data_from_list(bev_feat)
        coop_poses = self.stack_data_from_list(coop_poses)
        ego_poses = self.stack_data_from_list(requests, 'lidar_pose')
        transform_coop2ego = ego_poses.inverse() @ coop_poses
        dist_correction_matrix = utils.get_discretized_transformation_matrix(
            transform_coop2ego, self.discrete_ratio, self.downsample_rate)

        # transpose and flip to make the transformation correct
        x = rearrange(x, 'b c h w  -> b c w h')
        x = torch.flip(x, dims=(3,))
        # Only compensate non-ego vehicles
        B, C, H, W = x.shape

        T = utils.get_transformation_matrix(
            dist_correction_matrix.reshape(-1, 2, 3), (H, W))
        cav_features = utils.warp_affine(x.reshape(-1, C, H, W), T,
                                   (H, W))
        cav_features = cav_features.reshape(B, C, H, W)

        # flip and transpose back
        x = cav_features
        x = torch.flip(x, dims=(3,))
        x = rearrange(x, 'b c w h -> b c h w')

        bev_mask = utils.get_rotated_roi((B, 1, 1, H, W), T).squeeze(1)

        return {'bev_feat': x, 'bev_mask': bev_mask}

