from collections import OrderedDict
from cosense3d.config.pycfg.base import use_flash_attn, opv2vt, dairv2xt, hooks
from cosense3d.config.pycfg.template.petr_transformer import get_petr_transformer_cfg
from cosense3d.config.pycfg.template.spconv import get_spconv_cfg
from cosense3d.config.pycfg.template.query_guided_petr_head import get_query_guided_petr_head_cfg
from cosense3d.config.pycfg.template.det_anchor_dense import get_det_anchor_dense_cfg

voxel_size = [0.1, 0.1, 0.1]
out_stride = 8


def get_shared_modules(point_cloud_range, global_ref_time=0):
    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return OrderedDict(
        pts_backbone=get_spconv_cfg(
            gather_keys=['points'],
            scatter_keys=['voxel_feat', 'bev_feat', 'multi_scale_bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            bev_neck=dict(type='ssfa.SSFA', in_channels=64, out_channels=128,
                          shrink_strides=[2, 2], shrink_channels=[128, 256])
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['multi_scale_bev_feat'],
            scatter_keys=['det_local_dense', 'bev_local_dense'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_anchor_dense_cfg(
                    gather_keys=['bev_feat'],
                    scatter_keys=['detection'],
                    gt_keys=['global_bboxes_3d', 'global_labels_3d'],
                    in_channels=128,
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    stride=8,
                    pos_threshold=0.3,
                    neg_threshold=0.1,
                    score_thrshold=0.1,
                    get_boxes_when_training=True,
                ),
                dict(
                    type='heads.bev_dense.BevRoIDenseHead',
                    in_dim=256,
                )
            ],
            strides=[8, 32],
            losses=[True, False],
        ),

        keypoint_composer=dict(
            type='necks.cpm_composer.KeypointComposer',
            gather_keys=['det_local_dense', 'bev_feat', "voxel_feat", 'points'],
            scatter_keys=['keypoint_feat'],
            train_from_epoch=1,
            vsa=dict(
                type='vsa.VoxelSetAbstraction',
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                num_keypoints=2048,
                num_out_features=256,
                num_bev_features=128,
                num_rawpoint_features=3,
                enlarge_selection_boxes=True,
            )
        ),

        formatting = dict(
            type='necks.formatting.FPVRCNNToLTS',
            gather_keys=['multi_scale_bev_feat', 'keypoint_feat', 'bev_local_dense'],
            scatter_keys=['roi_local', 'roi_global', 'multi_scale_feat'],
            data_info=data_info,
            strides=[8, 32]
        ),

        temporal_fusion = dict(
            type='fusion.temporal_fusion.LocalTemporalFusion',
            gather_keys=['roi_local', 'roi_global', 'multi_scale_feat', 'memory'],
            scatter_keys=['temp_fusion_feat'],
            in_channels=256,
            ref_pts_stride=2,
            feature_stride=8,
            transformer_itrs=1,
            global_ref_time=global_ref_time,
            lidar_range=point_cloud_range,
            transformer=get_petr_transformer_cfg(use_flash_attn)
        ),

        spatial_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialQueryFusion',
            gather_keys=['temp_fusion_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            resolution=0.8
        ),

        det1_head = get_query_guided_petr_head_cfg(
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=False,
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
shared_modules_opv2vt = get_shared_modules(opv2vt.point_cloud_range, opv2vt.global_ref_time)

######################################################
#                     DairV2Xt
######################################################
test_hooks_dairv2xt = hooks.get_test_nms_eval_hooks(dairv2xt.point_cloud_range_test)
plots_dairv2xt = [hooks.get_detection_plot(dairv2xt.point_cloud_range_test)]
shared_modules_dairv2xt = get_shared_modules(dairv2xt.point_cloud_range, dairv2xt.global_ref_time)