from collections import OrderedDict

point_cloud_range = [-106, -40, -5.0, 106, 40, 3.0]
point_cloud_range_test = [-102.4, -38.4, -5.0, 102.4, 38.4, 3.0]


data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
            flip='xy',
            scale_ratio_range=[0.95, 1.05],
        ),
        pre_process=OrderedDict(
            remove_local_empty_boxes=dict(),
            remove_global_empty_boxes=dict(),
            sample_global_bev_tgt_pts=dict(sam_res=0.4, map_res=0.2, range=50, max_num_pts=5000, discrete=False)
        )
    ),
    test=dict(
        aug=dict()
    )
)


def get_v2vreal_cfg(seq_len, voxel_size, load_attributes=['xyz', 'intensity'], load_bev_map=False):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    pipeline_cpu = OrderedDict(
        LoadLidarPoints=dict(load_attributes=load_attributes),
        LoadAnnotations=dict(load3d_global=True, load3d_local=True, min_num_pts=3),
    )

    pipeline_cpu_test = OrderedDict(
        LoadLidarPoints=dict(load_attributes=load_attributes),
        LoadAnnotations=dict(load3d_global=True, load3d_local=True, min_num_pts=3),
    )

    inference_pipeline_cpu = OrderedDict(
        LoadLidarPoints=dict(load_attributes=load_attributes),
    )
    if load_bev_map:
        pipeline_cpu['LoadOPV2VBevMaps'] = dict(use_global_map=False, range=50, keys=['bev'])
        pipeline_cpu_test['LoadOPV2VBevMaps'] = dict(use_global_map=False, range=50, keys=['bev'])
        inference_pipeline_cpu['LoadOPV2VBevMaps'] = dict(use_global_map=True)

    return dict(
        name='v2vreal',
        dataset='temporal_cosense',
        meta_path='v2vreal',
        data_path='v2vreal',
        enable_split_sub_folder=False,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=7,
        com_range=200,
        seq_len=seq_len,
        train_pipeline=pipeline_cpu,
        test_pipeline=pipeline_cpu_test,
    )


seq4_pillar04 = get_v2vreal_cfg(4, [0.4, 0.4, 4])
seq4_vox04 = get_v2vreal_cfg(4, [0.4, 0.4, 0.4])
seq4_vox01 = get_v2vreal_cfg(4, [0.1, 0.1, 0.1])

seq1_vox02 = get_v2vreal_cfg(1, [0.2, 0.2, 0.2])
seq1_vox04 = get_v2vreal_cfg(1, [0.4, 0.4, 0.4])