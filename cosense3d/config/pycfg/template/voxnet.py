
def get_voxnet_cfg(gather_keys, scatter_keys, voxel_size, point_cloud_range,
                   neck=None, bev_compressor=None, sparse_cml=False):
    return dict(
            type='backbone3d.voxelnet.VoxelNet',
            gather_keys=gather_keys,
            scatter_keys=scatter_keys,
            voxel_generator=dict(
                type='voxel_generator.VoxelGenerator',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                max_points_per_voxel=32,
                max_voxels_train=32000,
                max_voxels_test=70000
            ),
            voxel_encoder=dict(
                type='pillar_encoder.PillarEncoder',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                features=['xyz', 'intensity', 'absolute_xyz'],
                channels=[64]
            ),
            cml=dict(type='voxnet_utils.CMLSparse' if sparse_cml else 'voxnet_utils.CML', in_channels=64),
            neck=neck,
            bev_compressor=bev_compressor
        )