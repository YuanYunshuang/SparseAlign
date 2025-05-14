import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg


voxel_size = [0.2, 0.2, 0.2]
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
        type='backbone3d.mink_unet.MinkUnet',
        gather_keys=['points'],
        scatter_keys=['pts_feat'],
        d=3,
        cache_strides=[2],
        in_dim=4,
        stride=out_stride,
        floor_height=point_cloud_range[2],
        data_info=data_info,
        height_compression=OrderedDict(p2=dict(channels=[96, 128, 128], steps=[5, 3])),
        enc_dim=32
    ),

    fusion = dict(
        type='fusion.naive_fusion.NaiveFusion',
        gather_keys=['pts_feat', 'received_response'],
        scatter_keys=['fused_feat'],
        stride=out_stride,
    ),

    fusion_neck = dict(
        type='necks.dilation_spconv.DilationSpconv',
        gather_keys=['fused_feat'],
        scatter_keys=['fused_neck_feat'],
        data_info=data_info,
        d=2,
        convs=dict(p2=dict(kernels=[5, 5, 3], in_dim=128, out_dim=128))
    ),

    bev_head = dict(
        type='heads.bev.BEV',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['bev'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        stride=out_stride,
        in_dim=128,
        num_cls=2,
        target_assigner=dict(type='target_assigners.BEVPointAssigner'),
        loss_cls=dict(type='EDLLoss', activation='exp', annealing_step=40, n_cls=2, loss_weight=1.0),
    ),

    detection_head = dict(
        type='heads.det_center_sparse.DetCenterSparse',
        gather_keys=['fused_neck_feat'],
        scatter_keys=['detection'],
        gt_keys=['global_bboxes_3d', 'global_labels_3d'],
        data_info=data_info,
        input_channels=128,
        shared_conv_channel=128,
        get_predictions=True,
        stride=out_stride,
        cls_head_cfg=dict(name='UnitedClsHead', one_hot_encoding=False),
        reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
        class_names_each_head=[['vehicle.car']],
        reg_channels=['box:6', 'dir:8', 'scr:4'],
        cls_assigner=dict(
            type='target_assigners.BEVCenternessAssigner',
            n_cls=1,
            min_radius=1.0,
            pos_neg_ratio=0,
            max_mining_ratio=0,
        ),
        box_assigner=dict(
            type='target_assigners.BoxCenterAssigner',
            voxel_size=voxel_size,
            lidar_range=point_cloud_range,
            stride=out_stride,
            detection_benchmark='Car',
            class_names_each_head=[['vehicle.car']],
            center_threshold=0.5,
            box_coder=dict(type='CenterBoxCoder'),
            activation='sigmoid',
            edl=False
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
    ),
)

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2v = hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_test)
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_test)]
shared_modules_opv2v = get_shared_modules(opv2v.point_cloud_range)
