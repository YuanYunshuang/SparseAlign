import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import use_flash_attn, opv2vt, dairv2xt, hooks
from cosense3d.config.pycfg.template.petr_transformer import get_petr_transformer_cfg
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg
from cosense3d.config.pycfg.template.query_guided_petr_head import get_query_guided_petr_head_cfg
from cosense3d.config.pycfg.template.det_center_sparse import get_det_center_sparse_cfg
from cosense3d.config.pycfg.template.bev_semseg_head import get_bev_semseg_head_cfg
from cosense3d.config.pycfg.template.rlseg_head import get_roadline_head_cfg
from cosense3d.config.pycfg.template.bev_head import get_bev_head_cfg, get_bev_multi_resolution_head_cfg


voxel_size = [0.4, 0.4, 0.4]
out_stride = 2


def get_shared_modules(point_cloud_range, global_ref_time=0, enc_dim=64):
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
            height_compression=[1, 2, 8],
            enc_dim=enc_dim,
        ),

        backbone_neck = dict(
            type='necks.dilation_spconv.DilationSpconv',
            gather_keys=['bev_feat'],
            scatter_keys=['bev_feat'],
            data_info=data_info,
            d=2,
            convs=dict(
                p1=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256),
                p2=dict(kernels=[3, 3, 3], in_dim=384, out_dim=256),
                p8=dict(kernels=[3, 3, 3], in_dim=256, out_dim=256)
            )
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['bev_feat'],
            scatter_keys=['roadline', 'bev_semseg_local', 'det_local', 'bev_local'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d', 'roadline_tgts'],
            heads=[
                get_roadline_head_cfg(data_info=data_info, stride=1),
                get_bev_semseg_head_cfg(
                    gt_keys=['bev_tgt_pts', 'local_bboxes_3d'],
                    semseg_head_type='heads.bev_semseg.EviSemsegHead',
                    data_info=data_info,
                    stride=1,
                    in_dim=256,
                    tgt_assigner_type='target_assigners.DiscreteBEVAssigner',
                ),
                get_det_center_sparse_cfg(
                    gt_keys=['local_bboxes_3d', 'local_labels_3d'],
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    in_channels=256,
                    generate_roi_scr=True,
                    cls_loss="FocalLoss",
                ),
                get_bev_head_cfg(
                    gt_keys=['local_bboxes_3d', 'local_labels_3d'],
                    data_info=data_info, out_stride=out_stride, in_dim=256, n_cls=2
                )
            ],
            strides=[1, 2, 8],
            losses=[True, True, False],
        ),

        temporal_fusion = dict(
            type='fusion.temporal_fusion.LocalTemporalFusion',
            gather_keys=['det_local', 'bev_local', 'bev_feat', 'memory'],
            scatter_keys=['temp_fusion_feat'],
            in_channels=256,
            ref_pts_stride=2,
            feature_stride=8,
            transformer_itrs=1,
            global_ref_time=global_ref_time,
            lidar_range=point_cloud_range,
            transformer=get_petr_transformer_cfg(use_flash_attn)
        ),

        spatial_alignment={},

        spatial_query_fusion=dict(
            type='fusion.spatial_query_fusion.SpatialQueryFusion',
            gather_keys=['temp_fusion_feat', 'received_response'],
            scatter_keys=['query_fusion_feat'],
            in_channels=256,
            pc_range=point_cloud_range,
            resolution=0.8
        ),

        spatial_bev_fusion=dict(
            type='fusion.naive_fusion.NaiveFusion',
            gather_keys=['bev_feat', 'received_response'],
            scatter_keys=['bev_fusion_feat'],
            stride=1
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
            gather_keys=['query_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        ),

        semseg_head=get_bev_semseg_head_cfg(
            gather_keys=['bev_fusion_feat'],
            scatter_keys=['bev_semseg'],
            gt_keys=['global_bev_tgt_pts', 'global_bboxes_3d'],
            semseg_head_type='heads.bev_semseg.GevSemsegHead',
            data_info=data_info,
            stride=out_stride,
            in_dim=256,
            tgt_assigner_type='target_assigners.ContiBEVAssigner',
        ),

    )

######################################################
#                     OPV2Vt
######################################################
test_hooks_opv2vt = hooks.get_test_nms_eval_hooks(opv2vt.point_cloud_range_test)
plots_opv2vt = [hooks.get_detection_plot(opv2vt.point_cloud_range_test)]
shared_modules_opv2vt = get_shared_modules(opv2vt.point_cloud_range, opv2vt.global_ref_time, enc_dim=64)

#--------- Ablation 1 : No RoI regression-------------
shared_modules_opv2vt_no_roi_reg = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_no_roi_reg['roi_head'] = get_bev_multi_resolution_head_cfg(
        gather_keys=['bev_feat'],
        scatter_keys=['bevseg_local'],
        gt_keys=['local_bboxes_3d', 'local_labels_3d'],
        data_info=dict(lidar_range=opv2vt.point_cloud_range, voxel_size=voxel_size),
    )

shared_modules_opv2vt_no_roi_reg['temporal_fusion'].update(
    type='fusion.temporal_fusion.LocalTemporalFusionV1',
    gather_keys=['bevseg_local', 'bev_feat', 'memory'],
)

#--------- Ablation 2 : No Timestamps for boxes --------
# see changes in yaml cfg of cav_prototype
shared_modules_opv2vt_no_t = copy.deepcopy(shared_modules_opv2vt)

#--------- Ablation 3 : No Global Attention ------------
shared_modules_opv2vt_no_global_attn = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_no_global_attn['temporal_fusion'].update(
    type='fusion.temporal_fusion.LocalTemporalFusionV2',
    gather_keys=['det_local', 'bev_feat', 'memory'],
)
Tlayer = shared_modules_opv2vt_no_global_attn['temporal_fusion']['transformer']['decoder']['transformerlayers']
Tlayer['attn_cfgs'] = Tlayer['attn_cfgs'][:1]
Tlayer['operation_order'] = ('self_attn', 'norm', 'ffn', 'norm')
shared_modules_opv2vt_no_global_attn['roi_head'] = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['bev_feat'],
            scatter_keys=['det_local'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_center_sparse_cfg(
                    voxel_size=voxel_size,
                    point_cloud_range=opv2vt.point_cloud_range,
                    in_channels=256,
                    generate_roi_scr=True
                ),
            ],
            strides=[2],
            losses=[True],
        )

#--------- Ablation 4 : Focal loss for RoI ------------
shared_modules_opv2vt_roi_focal_loss = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_roi_focal_loss['roi_head']['heads'][0] = get_det_center_sparse_cfg(
    voxel_size=voxel_size,
    point_cloud_range=opv2vt.point_cloud_range,
    in_channels=256,
    generate_roi_scr=True,
    cls_loss="FocalLoss"
)
shared_modules_opv2vt_roi_focal_loss['roi_head']['heads'][0]['cls_head_cfg'] = (
    dict(name='UnitedClsHead', one_hot_encoding=False))
shared_modules_opv2vt_roi_focal_loss['temporal_fusion']['norm_fusion'] = True

#--------- Comparative 1 : Pose error ------------
shared_modules_opv2vt_fcl_locerr = get_shared_modules(opv2vt.point_cloud_range, opv2vt.global_ref_time, 32)
shared_modules_opv2vt_fcl_locerr['spatial_alignment'] = dict(
    type='fusion.spatial_alignment.SpatialAlignment',
    gather_keys=['detection_local', 'received_response'],
    scatter_keys=['received_response'],
)

#--------- Comparative 2 : Latency ------------
shared_modules_opv2vt_fcl_lat = shared_modules_opv2vt_roi_focal_loss

######################################################
#                     DairV2Xt
######################################################
test_hooks_dairv2xt = hooks.get_test_nms_eval_hooks(dairv2xt.point_cloud_range_test)
plots_dairv2xt = [hooks.get_detection_plot(dairv2xt.point_cloud_range_test)]
shared_modules_dairv2xt = get_shared_modules(dairv2xt.point_cloud_range, dairv2xt.global_ref_time)

#--------- Ablation 1 : No RoI regression-------------
shared_modules_dairv2xt_no_roi_reg = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_no_roi_reg['roi_head'] = get_bev_multi_resolution_head_cfg(
        gather_keys=['bev_feat'],
        scatter_keys=['bevseg_local'],
        gt_keys=['local_bboxes_3d', 'local_labels_3d'],
        data_info=dict(lidar_range=dairv2xt.point_cloud_range, voxel_size=voxel_size),
    )
shared_modules_dairv2xt_no_roi_reg['temporal_fusion'].update(
    type='fusion.temporal_fusion.LocalTemporalFusionV1',
    gather_keys=['bevseg_local', 'bev_feat', 'memory'],
)

#--------- Ablation 2 : No Timestamps for boxes --------
# see changes in yaml cfg of cav_prototype
shared_modules_dairv2xt_no_t = shared_modules_dairv2xt

#--------- Ablation 3 : No Global Attention ------------
shared_modules_dairv2xt_no_global_attn = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_no_global_attn['temporal_fusion'].update(
    type='fusion.temporal_fusion.LocalTemporalFusionV2',
    gather_keys=['det_local', 'bev_feat', 'memory'],
)
Tlayer = shared_modules_dairv2xt_no_global_attn['temporal_fusion']['transformer']['decoder']['transformerlayers']
Tlayer['attn_cfgs'] = Tlayer['attn_cfgs'][:1]
Tlayer['operation_order'] = ('self_attn', 'norm', 'ffn', 'norm')
shared_modules_dairv2xt_no_global_attn['roi_head'] = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['bev_feat'],
            scatter_keys=['det_local'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            heads=[
                get_det_center_sparse_cfg(
                    voxel_size=voxel_size,
                    point_cloud_range=opv2vt.point_cloud_range,
                    in_channels=256,
                    generate_roi_scr=True
                ),
            ],
            strides=[2],
            losses=[True],
        )

#--------- Ablation 4 : Focal loss for RoI ------------
shared_modules_dairv2xt_roi_focal_loss = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_roi_focal_loss['roi_head']['heads'][0] = get_det_center_sparse_cfg(
    voxel_size=voxel_size,
    point_cloud_range=opv2vt.point_cloud_range,
    in_channels=256,
    generate_roi_scr=True,
    cls_loss="FocalLoss"
)
shared_modules_dairv2xt_roi_focal_loss['roi_head']['heads'][0]['cls_head_cfg'] = (
    dict(name='UnitedClsHead', one_hot_encoding=False))

#--------- Ablation 5 : Focal loss and Gaussian GT for RoI ------------
shared_modules_dairv2xt_roi_focal_loss_gaussian = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_roi_focal_loss_gaussian['roi_head']['heads'][0] = get_det_center_sparse_cfg(
    voxel_size=voxel_size,
    point_cloud_range=opv2vt.point_cloud_range,
    in_channels=256,
    generate_roi_scr=True,
    cls_loss="FocalLoss",
    use_gaussian=True,
    sigma=1.0
)
shared_modules_dairv2xt_roi_focal_loss_gaussian['roi_head']['heads'][0]['cls_head_cfg'] = (
    dict(name='UnitedClsHead', one_hot_encoding=False))

