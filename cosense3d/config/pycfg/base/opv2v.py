from collections import OrderedDict

# point_cloud_range = [-144, -51.2, -3.0, 144, 51.2, 1.0]
point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
# point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
point_cloud_range_test = [-140, -40, -3.0, 140, 40, 1.0]


data_manager = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57 / 2, 1.57 / 2],
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


data_manager_no_bev_tgt = dict(
    train=dict(
        aug=dict(
            rot_range=[-1.57, 1.57],
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

def get_opv2v_cfg(seq_len, voxel_size, latency=0, load_attributes=['xyz', 'intensity'],
                  load_bev_map=False, max_num_cavs=7, loc_err=[0., 0., 0.]):
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
        pipeline_cpu['LoadOPV2VBevMaps'] = dict(use_global_map=True, range=50, keys=['bev'])
        pipeline_cpu_test['LoadOPV2VBevMaps'] = dict(use_global_map=True, range=50, keys=['bev'])
        inference_pipeline_cpu['LoadOPV2VBevMaps'] = dict(use_global_map=True)

    return dict(
        name='opv2v',
        dataset='temporal_cosense',
        meta_path='opv2v',
        data_path='opv2v',
        enable_split_sub_folder=False,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=max_num_cavs,
        com_range=70,
        latency=latency,
        seq_len=seq_len,
        n_loss_frame=1,
        train_pipeline=pipeline_cpu,
        test_pipeline=pipeline_cpu_test,
        loc_err=loc_err
    )


seq4_pillar04 = get_opv2v_cfg(4, [0.4, 0.4, 4])
seq4_vox04 = get_opv2v_cfg(4, [0.4, 0.4, 0.4])
seq4_vox02 = get_opv2v_cfg(4, [0.4, 0.4, 0.4])
seq4_vox01 = get_opv2v_cfg(4, [0.1, 0.1, 0.1])
seq1_vox04 = get_opv2v_cfg(1, [0.4, 0.4, 0.4])
seq1_vox02 = get_opv2v_cfg(1, [0.2, 0.2, 0.2])
seq1_vox01 = get_opv2v_cfg(1, [0.1, 0.1, 0.1])
seq1_vox040460 = get_opv2v_cfg(1, [0.4, 0.4, 6.0])

seq4_pillar04_bevmap = get_opv2v_cfg(4, [0.4, 0.4, 4], load_bev_map=True)
seq1_vox02_bevmap = get_opv2v_cfg(1, [0.2, 0.2, 0.2],
                                  load_attributes=['xyz', 'intensity', 'distance', 'cosine', 'sine'],
                                  load_bev_map=True)
seq1_vox04_bevmap = get_opv2v_cfg(1, [0.4, 0.4, 0.4], load_bev_map=True)
seq1_vox04_bevmap_ego_only = get_opv2v_cfg(1, [0.4, 0.4, 0.4], load_bev_map=True, max_num_cavs=1)