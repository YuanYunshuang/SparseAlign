import copy
from collections import OrderedDict

import numpy as np

# point_cloud_range = [-144, -144, -3.0, 144, 144, 1.0]
point_cloud_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
point_cloud_range_test = [-140, -40, -3.0, 140, 40, 1.0]
global_ref_time = 0.05

pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
    LoadAnnotations=dict(load3d_global=True, load3d_local=True,
                         load_global_time=True, with_velocity=True, min_num_pts=0),
)

inference_pipeline_cpu = OrderedDict(
    LoadLidarPoints=dict(load_attributes=['xyz', 'intensity', 'time']),
    LoadAnnotations=dict(),
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
        aug=dict(),
        # pre_process = ['remove_local_empty_boxes',
        #                'remove_global_empty_boxes']
    )
)


def get_opv2vt_cfg(seq_len, voxel_size, latency=0, load_bevmap=False,
                   load_roadline=False, loc_err=[0., 0., 0.]):
    """
    Examples of setting loc. errors:

    .. highlight:: python
    .. code-block:: python

        # introduce loc. errors in the dataloader will lead to different errors at different frames
        pipeline_cpu['LoadAnnotations']['loc_err'] = loc_err
        inference_pipeline_cpu['LoadAnnotations']['loc_err'] = loc_err

        # instead, one can introduce unified errors for a short sequence by setting the data_manager argument loc_err
        data_manager['test']['loc_err'] = loc_err
    """
    train_pipeline = copy.deepcopy(pipeline_cpu)
    inf_pipeline = copy.deepcopy(inference_pipeline_cpu)
    if load_bevmap:
        train_pipeline['LoadOPV2VBevMaps'] = dict()
    if load_roadline:
        train_pipeline['LoadCarlaRoadlineMaps'] = dict(ego_only=False)
        inf_pipeline['LoadCarlaRoadlineMaps'] = dict(ego_only=False)
    # if loc_err[0] > 0:
    #     train_pipeline['GenerateErrorPose'] = dict()

    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
        name='opv2vt',
        dataset='temporal_cosense',
        meta_path='opv2vt',
        data_path='opv2vt',
        enable_split_sub_folder=False,
        DetectionBenchmark='Car',
        data_info=data_info,
        lidar_range=point_cloud_range,
        voxel_size=voxel_size,
        batch_size_train=4,
        batch_size_test=1,
        n_workers=4,
        max_num_cavs=7,
        com_range=70,
        latency=latency,
        seq_len=seq_len,
        n_loss_frame=1,
        train_pipeline=train_pipeline,
        test_pipeline=train_pipeline,
        inf_pipeline=inf_pipeline,
        loc_err=loc_err,
    )


seq4_pillar04 = get_opv2vt_cfg(4, [0.4, 0.4, 4])
seq4_vox04 = get_opv2vt_cfg(4, [0.4, 0.4, 0.4])
seq4_vox04_lat1 = get_opv2vt_cfg(4, [0.4, 0.4, 0.4], 1)
seq4_vox04_lat2 = get_opv2vt_cfg(4, [0.4, 0.4, 0.4], 2)
seq4_vox04_randlat = get_opv2vt_cfg(4, [0.4, 0.4, 0.4], -1)
seq4_vox04_randlat_rl = get_opv2vt_cfg(4, [0.4, 0.4, 0.4], -1, load_bevmap=True)
seq1_vox04_randlat_rl = get_opv2vt_cfg(1, [0.4, 0.4, 0.4], -1, load_bevmap=True)
seq4_vox01 = get_opv2vt_cfg(4, [0.1, 0.1, 0.1])
seq1_vox04 = get_opv2vt_cfg(1, [0.4, 0.4, 0.4])

div = 1
seq4_vox04_locerr_rl = get_opv2vt_cfg(4, [0.4, 0.4, 0.4],
                                   load_roadline=True,
                                   loc_err=[0.5 / div, 0.5 / div, 0.0174533 / div])
                                   # loc_err=[0.5, 0.5, 0.0872665])

seq4_vox04_locerr = get_opv2vt_cfg(4, [0.4, 0.4, 0.4],
                                   loc_err=[1.0 / div, 1.0 / div, np.deg2rad(10) / div])