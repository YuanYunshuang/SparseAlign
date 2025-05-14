import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg


voxel_size = [0.4, 0.4, 0.4]
out_stride = 1


def get_shared_modules(point_cloud_range, enc_dim=32):
    """
    gather_keys:
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys:
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
        pts_backbone=get_minkunet_cfg(
            gather_keys=['points'],
            scatter_keys=['bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            kernel_size_layer1=5,
            cache_strides=[1],
            height_compression=[1],
            enc_dim=enc_dim,
        ),

        backbone_neck = dict(
            type='necks.dilation_spconv.DilationSpconv',
            gather_keys=['bev_feat'],
            scatter_keys=['bev_feat'],
            data_info=data_info,
            d=2,
            convs=dict(
                p1=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256)
            )
        ),

        rlseg_head = dict(
            type='heads.bev_roadline.BEVRoadLine',
            gather_keys=['bev_feat'],
            scatter_keys=['roadline'],
            gt_keys=['roadline_tgts'],
            data_info=data_info,
            stride=1,
            in_dim=256,
            target_assigner=dict(type='target_assigners.RoadLineAssigner', res=0.4, range=50),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        )

    )

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2v = hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_bev_test)
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_bev_test)]
shared_modules_opv2v = get_shared_modules(opv2v.point_cloud_range_bev, enc_dim=64)
