
def get_spconv_cfg(gather_keys, scatter_keys, voxel_size, point_cloud_range,
                   in_channels=4, out_channels=64, bev_neck=None, bev_compressor=None):
    return dict(
        type='backbone3d.spconv.Spconv',
        gather_keys=gather_keys,
        scatter_keys=scatter_keys,
        in_channels=in_channels,
        out_channels=out_channels,
        voxel_generator=dict(
            type='voxel_generator.VoxelGenerator',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            max_points_per_voxel=32,
            max_voxels_train=32000,
            max_voxels_test=70000
        ),
        voxel_encoder=dict(
            type='voxel_encoder.MeanVFE',
            num_point_features=in_channels,
        ),
        bev_neck=bev_neck,
        bev_compressor=bev_compressor
    )