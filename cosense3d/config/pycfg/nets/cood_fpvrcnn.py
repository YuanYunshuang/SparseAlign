import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg


voxel_size = [0.1, 0.1, 0.1]
out_stride = 2


def get_shared_modules(point_cloud_range):
    """
    gather_keys:
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys:
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
    pts_backbone = dict(
        type='backbone3d.spconv.Spconv',
        gather_keys=['points'],
        scatter_keys=['voxel_feat', 'bev_feat'],
        in_channels=4,
        out_channels=64,
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
            num_point_features=4,
        ),
        bev_neck=dict(type='ssfa.SSFA', in_channels=64, out_channels=128),
    ),

    detection_head_local = dict(
        type='heads.det_anchor_dense.DetAnchorDense',
        gather_keys=['bev_feat'],
        scatter_keys=['detection_local'],
        gt_keys=['local_bboxes_3d', 'local_labels_3d'],
        in_channels=128,
        get_boxes_when_training=True,
        target_assigner=dict(
            type='target_assigners.BoxAnchorAssigner',
            box_size=[3.9, 1.6, 1.56],
            dirs=[0, 90],
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=8,
            pos_threshold=0.6,
            neg_threshold=0.45,
            score_thrshold=0.15,
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        ),
        loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                        gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_box = dict(type='SmoothL1Loss', loss_weight=2.0),
    ),

    keypoint_composer=dict(
        type='necks.cpm_composer.KeypointComposer',
        gather_keys=['detection_local', 'bev_feat', "voxel_feat", 'points'],
        scatter_keys=['keypoint_feat'],
        train_from_epoch=5,
        vsa=dict(
            type='vsa.VoxelSetAbstraction',
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_keypoints=4096,
            num_out_features=32,
            num_bev_features=128,
            num_rawpoint_features=3,
            enlarge_selection_boxes=True,
        )
    ),

    fusion=dict(
        type='fusion.keypoints.KeypointsFusion',
        gather_keys=['keypoint_feat', 'received_response'],
        scatter_keys=['keypoint_feat_fused'],
        train_from_epoch=5,
        lidar_range=point_cloud_range,
    ),

    detection_head_global = dict(
        type='heads.det_roi_refine.KeypointRoIHead',
        gather_keys=['keypoint_feat_fused'],
        scatter_keys=['detection'],
        gt_keys=['global_bboxes_3d'],
        train_from_epoch=5,
        num_cls=1,
        in_channels=32,
        n_fc_channels=256,
        dp_ratio=0.3,
        roi_grid_pool=dict(
            grid_size=6,
            mlps=[[64, 64], [64, 64]],
            pool_radius=[0.8, 1.6],
            n_sample=[16, 16],
            pool_method='max_pool',
        ),
        target_assigner=dict(
            type='target_assigners.RoIBox3DAssigner',
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        )
    )
)

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2v = hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_test)
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_test)]
shared_modules_opv2v = get_shared_modules([-140.8, -41.6, -3, 140.8, 41.6, 1])
