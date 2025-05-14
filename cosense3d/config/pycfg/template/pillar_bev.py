from collections import OrderedDict
from cosense3d.config import add_cfg_keys


@add_cfg_keys
def get_pillar_bev_cfg(voxel_size, point_cloud_range, **kwargs):
    return dict(
            type='backbone3d.pillar_bev.PillarBEV',
            in_channels=64,
            layer_nums=[3, 5, 8],
            layer_strides=[2, 2, 2],
            downsample_channels=[64, 128, 256],
            upsample_strides=[1, 2, 4],
            upsample_channels=[128, 128, 128],
            voxel_generator=dict(
                type='voxel_generator.VoxelGenerator',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                max_points_per_voxel=32,
                max_voxels_train=32000,
                max_voxels_test=70000
            ),
            pillar_encoder=dict(
                type='pillar_encoder.PillarEncoder',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                features=['xyz', 'intensity', 'absolute_xyz'],
                channels=[64]
            ),
            bev_shrinker=dict(
                type='downsample_conv.DownsampleConv',
                in_channels=384,  # 128 * 3
                dims=[256]
            ),
        )