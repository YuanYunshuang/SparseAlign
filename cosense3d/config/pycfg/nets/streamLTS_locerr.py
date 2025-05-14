import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import use_flash_attn, opv2vt, dairv2xt, hooks
from cosense3d.config.pycfg.template.petr_transformer import get_petr_transformer_cfg
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg
from cosense3d.config.pycfg.template.query_guided_petr_head import get_query_guided_petr_head_cfg
from cosense3d.config.pycfg.template.det_center_sparse import get_det_center_sparse_cfg
from cosense3d.config.pycfg.template.bev_head import get_bev_head_cfg, get_bev_multi_resolution_head_cfg
from cosense3d.config.pycfg.template.rlseg_head import get_roadline_head_cfg


voxel_size = [0.4, 0.4, 0.4]
out_stride = 2


def get_shared_modules(point_cloud_range, global_ref_time=0, enc_dim=32):
    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
        pts_backbone=get_minkunet_cfg(
            freeze=False,
            gather_keys=['points'],
            scatter_keys=['bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            kernel_size_layer1=3,
            height_compression=[2, 8],
            enc_dim=enc_dim,
        ),

        rl_backbone=get_minkunet_cfg(
            freeze=True,
            gather_keys=['points_rl'],
            scatter_keys=['bev_feat_rl'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            kernel_size_layer1=5,
            cache_strides=[1],
            height_compression=[1],
            enc_dim=enc_dim,
        ),

        backbone_neck = dict(
            type='necks.dilation_spconv.DilationSpconv',
            freeze=False,
            gather_keys=['bev_feat'],
            scatter_keys=['bev_feat'],
            data_info=data_info,
            d=2,
            convs=dict(
                p2=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256),
                p8=dict(kernels=[3, 3, 3], in_dim=256, out_dim=256)
            )
        ),

        rl_neck=dict(
            type='necks.dilation_spconv.DilationSpconv',
            freeze=True,
            gather_keys=['bev_feat_rl'],
            scatter_keys=['bev_feat_rl'],
            data_info=data_info,
            d=2,
            convs=dict(
                p1=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256)
            )
        ),

        rlseg_head=dict(
            type='heads.bev_roadline.BEVRoadLine',
            freeze=True,
            gather_keys=['bev_feat_rl'],
            scatter_keys=['roadline_pred'],
            gt_keys=['roadline_tgts'],
            data_info=data_info,
            stride=1,
            in_dim=256,
            target_assigner=dict(type='target_assigners.RoadLineAssigner', res=0.4, range=50),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                          gamma=2.0, alpha=0.25, loss_weight=2.0),
        ),

        localization=dict(
            type='necks.spatial_alignment.MapRegistration',
            freeze=True,
            gather_keys=['roadline_pred', 'roadline', 'lidar_poses', 'lidar_poses_gt'],
            scatter_keys=['lidar_poses_corrected', 'roadline_pred'],
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            freeze=False,
            gather_keys=['bev_feat'],
            scatter_keys=['det_local', 'bev_local'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_center_sparse_cfg(
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    in_channels=256,
                    generate_roi_scr=True,
                    cls_assigner='BEVBoxAssigner',
                    cls_loss="FocalLoss"
                ),
                get_bev_head_cfg(
                    data_info, out_stride, in_dim=256, n_cls=2
                )
            ],
            strides=[2, 8],
            losses=[True, False],
        ),

        temporal_fusion = dict(
            type='fusion.temporal_fusion.LocalTemporalFusion',
            freeze=False,
            gather_keys=['det_local', 'bev_local', 'bev_feat', 'memory'],
            scatter_keys=['temp_fusion_feat'],
            in_channels=256,
            ref_pts_stride=2,
            feature_stride=8,
            transformer_itrs=1,
            global_ref_time=global_ref_time,
            lidar_range=point_cloud_range,
            transformer=get_petr_transformer_cfg(use_flash_attn),
            norm_fusion=False,
        ),

        det1_head = get_query_guided_petr_head_cfg(
            freeze=False,
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=False,
            pred_while_training=True
        ),

        spatial_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialQueryAlignFusionRL',
            gather_keys=['detection_local', 'roadline', 'roadline_pred', 'temp_fusion_feat',
                         'lidar_poses_corrected', 'lidar_poses', 'lidar_pose_aug',
                         'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            resolution=0.8
        ),

        det2_head = get_query_guided_petr_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        ),

    )

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2vt = hooks.get_test_nms_eval_hooks(opv2vt.point_cloud_range_test)
plots_opv2vt = [hooks.get_detection_plot(opv2vt.point_cloud_range_test)]
shared_modules_opv2vt = get_shared_modules(opv2vt.point_cloud_range, opv2vt.global_ref_time, enc_dim=64)



######################################################
#                     DairV2Xt
######################################################
test_hooks_dairv2xt = hooks.get_test_nms_eval_hooks(dairv2xt.point_cloud_range_test)
plots_dairv2xt = [hooks.get_detection_plot(dairv2xt.point_cloud_range_test)]
shared_modules_dairv2xt = get_shared_modules(dairv2xt.point_cloud_range, dairv2xt.global_ref_time)


