from collections import OrderedDict
from cosense3d.config.pycfg.base import use_flash_attn, opv2vt, dairv2xt, hooks
from cosense3d.config.pycfg.template.petr_transformer import get_transformer_cfg
from cosense3d.config.pycfg.template.voxnet import get_voxnet_cfg
from cosense3d.config.pycfg.template.query_det_head import get_query_det_head_cfg
from cosense3d.config.pycfg.template.det_anchor_dense import get_det_anchor_dense_cfg

voxel_size = [0.4, 0.4, 0.4]
out_stride = 2


def get_shared_modules(point_cloud_range, global_ref_time=0.0):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
      pts_backbone=get_voxnet_cfg(
            gather_keys=['points'],
            scatter_keys=['bev_feat', 'multi_scale_bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            sparse_cml=True,
            neck=dict(type='bev_rpn.CustomRPN', out_channels=256, num_layers=2),
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['multi_scale_bev_feat'],
            scatter_keys=['det_local_dense'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_anchor_dense_cfg(
                    gather_keys=['bev_feat'],
                    scatter_keys=['detection'],
                    gt_keys=['global_bboxes_3d', 'global_labels_3d'],
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    pos_threshold=0.3,
                    neg_threshold=0.1,
                ),
            ],
            strides=[2],
            losses=[True],
        ),

        formatting = dict(
            type='necks.formatting.DenseToSparse',
            gather_keys=['multi_scale_bev_feat', 'det_local_dense'],
            scatter_keys=['multi_scale_bev_feat', 'det_local_sparse'],
            data_info=data_info,
            strides=[2]
        ),

        temporal_fusion=dict(
            type='fusion.temporal_fusion_v2.LocalTemporalFusion',
            gather_keys=['det_local_sparse', 'multi_scale_bev_feat', 'memory'],
            scatter_keys=['temp_fusion_feat'],
            # freeze=True,
            in_channels=256,
            ref_pts_stride=2,
            transformer_itrs=1,
            global_ref_time=global_ref_time,
            lidar_range=point_cloud_range,
            transformer=get_transformer_cfg(use_flash_attn),
        ),

        spatial_alignment=dict(
            type='fusion.pose_align.NbrBasedPoseAlign',
            # type='fusion.pose_align.PseudoPoseAlign',
            gather_keys=['T_aug2g', 'received_response', 'detection_local'],
            scatter_keys=['T_c2e'],
            gt_keys=[],
            disable_align=True
        ),

        spatial_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialAlignment',
            gather_keys=['T_c2e', 'temp_fusion_feat', 'received_response', 'detection_local', 'global_bboxes_3d'],
            scatter_keys=['spatial_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            stride=out_stride,
        ),

        det1_head=get_query_det_head_cfg(
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            # freeze=True,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
            cls_assigner='BEVBoxAssigner',
        ),

        det2_head=get_query_det_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
            cls_assigner='BEVBoxAssigner'
        ),

    )


######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2vt = hooks.get_test_nms_eval_hooks(opv2vt.point_cloud_range_test)
plots_opv2vt = [hooks.get_detection_plot(opv2vt.point_cloud_range_test)]
shared_modules_opv2vt = get_shared_modules(opv2vt.point_cloud_range, opv2vt.global_ref_time)

######################################################
#                     DairV2Xt
######################################################
test_hooks_dairv2xt = hooks.get_test_nms_eval_hooks(dairv2xt.point_cloud_range_test)
plots_dairv2xt = [hooks.get_detection_plot(dairv2xt.point_cloud_range_test)]
shared_modules_dairv2xt = get_shared_modules(dairv2xt.point_cloud_range, dairv2xt.global_ref_time)