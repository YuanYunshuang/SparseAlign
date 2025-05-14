import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg


voxel_size = [0.4, 0.4, 0.4]
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
        type='backbone3d.voxelnet.VoxelNet',
        gather_keys=['points'],
        scatter_keys=['bev_feat'],
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
        cml=dict(type='voxnet_utils.CML', in_channels=64),
    ),

    fusion = dict(
        type='fusion.attn_fusion.DenseAttentionFusion',
        gather_keys=['bev_feat', 'received_response'],
        scatter_keys=['bev_feat_fused'],
        feature_dim=128,
        neck=dict(type='bev_rpn.RPN', anchor_num=2)
    ),

    detection_head = dict(
        type='heads.det_anchor_dense.DetAnchorDense',
        gather_keys=['bev_feat_fused'],
        scatter_keys=['detection'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        in_channels=768,
        target_assigner=dict(
            type='target_assigners.BoxAnchorAssigner',
            box_size=[3.9, 1.6, 1.56],
            dirs=[0, 90],
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=2,
            pos_threshold=0.6,
            neg_threshold=0.45,
            score_thrshold=0.25,
            box_coder=dict(type='ResidualBoxCoder', mode='simple_dist')
        ),
        loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                        gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_box = dict(type='SmoothL1Loss', loss_weight=2.0),
    )
)

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2v = hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_test)
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_test)]
shared_modules_opv2v = get_shared_modules([-140.8, -41.6, -3, 140.8, 41.6, 1])
