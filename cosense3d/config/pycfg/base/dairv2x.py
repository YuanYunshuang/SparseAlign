import copy
from collections import OrderedDict

point_cloud_range = [-102.4, -41.6, -3.0, 102.4, 41.6, 1.0]
# point_cloud_range_test = [-100, -38.4, -3.0, 100, 38.4, 1.0]
point_cloud_range_test = [-100, -40, -3.0, 100, 40, 1.0]
global_ref_time = 0.0

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity'], time_offset=1.6261*1e9),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         with_velocity=True, min_num_pts=3, load_global_time=True),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity']),
)

data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57 / 2, 1.57 / 2],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=['remove_local_empty_boxes',
                     'remove_global_empty_boxes']
    ),
    test=dict(
        aug=dict()
    )
)


def get_dairv2x_cfg(seq_len, voxel_size, latency=0, load_bevmap=False):
    pipeline = copy.deepcopy(pipeline_cpu)
    if load_bevmap:
        pipeline['LoadOPV2VBevMaps'] = dict()
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
        name='dairv2x',
        dataset='temporal_cosense',
        meta_path='dairv2x',
        data_path='dairv2x',
        enable_split_sub_folder=True,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=2,
        com_range=200,
        latency=latency,
        seq_len=seq_len,
        train_pipeline=pipeline,
        test_pipeline=pipeline,
    )


seq4_pillar04 = get_dairv2x_cfg(4, [0.4, 0.4, 4])
seq4_vox04 = get_dairv2x_cfg(4, [0.4, 0.4, 0.4])
seq1_vox04 = get_dairv2x_cfg(1, [0.4, 0.4, 0.4])
seq4_vox04_lat1 = get_dairv2x_cfg(4, [0.4, 0.4, 0.4], 1)
seq4_vox01 = get_dairv2x_cfg(4, [0.1, 0.1, 0.1])
seq4_vox04_randlat = get_dairv2x_cfg(4, [0.4, 0.4, 0.4], -1)