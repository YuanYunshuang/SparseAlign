import copy
from collections import OrderedDict

from cosense3d.config.pycfg.base import use_flash_attn, opv2vt, dairv2xt, hooks
from cosense3d.config.pycfg.template.petr_transformer import get_transformer_cfg
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg
from cosense3d.config.pycfg.template.query_det_head import get_query_det_head_cfg, get_query_det_second_head_cfg
from cosense3d.config.pycfg.template.det_center_sparse import get_det_center_sparse_cfg
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
    unet_dims = {1: enc_dim, 2: enc_dim * 3, 4: enc_dim * 4, 8: enc_dim * 4}
    return OrderedDict(
        pts_backbone=dict(
            type='backbone3d.unet.Unet',
            gather_keys=['points'],
            scatter_keys=['bev_feat'],
            # freeze=True,
            data_info=data_info,
            stride=2,
            in_dim=4,
            # height_attn=dict(p2=[unet_dims[2], 256]),
            cache_strides=[2],
            enc_dim=enc_dim,
            floor_height=-3.,
        ),

        roi_head = dict(
            type='heads.multitask_head.MultiTaskHead',
            gather_keys=['bev_feat'],
            scatter_keys=['det_local'],
            gt_keys=['local_bboxes_3d', 'local_labels_3d'],
            # freeze=True,
            heads=[
                get_det_center_sparse_cfg(
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    in_channels=256,
                    generate_roi_scr=True,
                    cls_assigner='BEVBoxAssigner',
                    cls_loss="FocalLoss"
                ),
            ],
            strides=[2],
            losses=[True],
        ),

        temporal_fusion = dict(
            type='fusion.temporal_fusion_v2.LocalNaiveFusion',
            gather_keys=['det_local', 'bev_feat', 'memory'],
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

        det1_head = get_query_det_head_cfg(
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

        det2_head = get_query_det_head_cfg(
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

#--------- Ablation 1: no expand 2d, no expand 3d --------
shared_modules_opv2vt_no_expand = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_no_expand['pts_backbone']['expand_2d'] = False
shared_modules_opv2vt_no_expand['pts_backbone']['expand_3d'] = False
#--------- Ablation 2: no expand 3d --------
shared_modules_opv2vt_no_expand_3d = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_no_expand_3d['pts_backbone']['expand_3d'] = False
#--------- Ablation 4: dilconv --------
shared_modules_opv2vt_dilconv = copy.deepcopy(shared_modules_opv2vt)
shared_modules_opv2vt_dilconv['pts_backbone']['expand_2d'] = False
shared_modules_opv2vt_dilconv['pts_backbone']['expand_3d'] = False
shared_modules_opv2vt_dilconv['pts_backbone']['type'] = 'backbone3d.unet.UnetDilConv'
#--------- Ablation 7: simple dist box coder --------
shared_modules_opv2vt_simple_dist = copy.deepcopy(shared_modules_opv2vt_no_expand)
shared_modules_opv2vt_simple_dist['roi_head']['heads'][0]['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_simple_dist['roi_head']['heads'][0]['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_opv2vt_simple_dist['roi_head']['heads'][0]['reg_channels'] = ['box:6', 'dir:1', 'scr:0']
shared_modules_opv2vt_simple_dist['det1_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_simple_dist['det1_head']['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_opv2vt_simple_dist['det1_head']['reg_channels'] = ['box:6', 'dir:1', 'scr:0', 'vel:2']
shared_modules_opv2vt_simple_dist['det2_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_simple_dist['det2_head']['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_opv2vt_simple_dist['det2_head']['reg_channels'] = ['box:6', 'dir:1', 'scr:0', 'vel:2']
#--------- Ablation 7: simple dist box coder  and second sin encoding--------
shared_modules_opv2vt_simple_dist_second = copy.deepcopy(shared_modules_opv2vt_simple_dist)
shared_modules_opv2vt_simple_dist_second['roi_head']['heads'][0]['sin_enc'] = True
shared_modules_opv2vt_simple_dist_second['det1_head']['sin_enc'] = True
shared_modules_opv2vt_simple_dist_second['det2_head']['sin_enc'] = True
#--------- Ablation 10: sin cos box coder --------
shared_modules_opv2vt_sin_cos = copy.deepcopy(shared_modules_opv2vt_no_expand)
shared_modules_opv2vt_sin_cos['roi_head']['heads'][0]['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_sin_cos['roi_head']['heads'][0]['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_opv2vt_sin_cos['roi_head']['heads'][0]['reg_channels'] = ['box:6', 'dir:2', 'scr:0']
shared_modules_opv2vt_sin_cos['det1_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_sin_cos['det1_head']['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_opv2vt_sin_cos['det1_head']['reg_channels'] = ['box:6', 'dir:2', 'scr:0', 'vel:2']
shared_modules_opv2vt_sin_cos['det2_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_opv2vt_sin_cos['det2_head']['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_opv2vt_sin_cos['det2_head']['reg_channels'] = ['box:6', 'dir:2', 'scr:0', 'vel:2']


######################################################
#                     DairV2Xt
######################################################
test_hooks_dairv2xt = hooks.get_test_nms_eval_hooks(dairv2xt.point_cloud_range_test)
plots_dairv2xt = [hooks.get_detection_plot(dairv2xt.point_cloud_range_test)]
shared_modules_dairv2xt = get_shared_modules(dairv2xt.point_cloud_range, dairv2xt.global_ref_time)

#--------- Ablation 1: no expand 2d, no expand 3d --------
shared_modules_dairv2xt_no_expand = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_no_expand['pts_backbone']['expand_2d'] = False
shared_modules_dairv2xt_no_expand['pts_backbone']['expand_3d'] = False
#--------- Ablation 2: no expand 3d --------
shared_modules_dairv2xt_no_expand_3d = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_no_expand_3d['pts_backbone']['expand_3d'] = False
#--------- Ablation 3: no tempfusion --------
shared_modules_dairv2xt_no_tempfusion = copy.deepcopy(shared_modules_dairv2xt)
#--------- Ablation 4: dilconv --------
shared_modules_dairv2xt_dilconv = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_dilconv['pts_backbone']['expand_2d'] = False
shared_modules_dairv2xt_dilconv['pts_backbone']['expand_3d'] = False
shared_modules_dairv2xt_dilconv['pts_backbone']['type'] = 'backbone3d.unet.UnetDilConv'
#--------- Ablation 7: simple dist box coder --------
shared_modules_dairv2xt_simple_dist = copy.deepcopy(shared_modules_dairv2xt_no_expand)
shared_modules_dairv2xt_simple_dist['roi_head']['heads'][0]['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_simple_dist['roi_head']['heads'][0]['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_dairv2xt_simple_dist['roi_head']['heads'][0]['reg_channels'] = ['box:6', 'dir:1', 'scr:0']
shared_modules_dairv2xt_simple_dist['det1_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_simple_dist['det1_head']['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_dairv2xt_simple_dist['det1_head']['reg_channels'] = ['box:6', 'dir:1', 'scr:0', 'vel:2']
shared_modules_dairv2xt_simple_dist['det2_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_simple_dist['det2_head']['box_assigner']['box_coder']['mode'] = 'simple'
shared_modules_dairv2xt_simple_dist['det2_head']['reg_channels'] = ['box:6', 'dir:1', 'scr:0', 'vel:2']
#--------- Ablation 7: simple dist box coder  and second sin encoding--------
shared_modules_dairv2xt_simple_dist_second = copy.deepcopy(shared_modules_dairv2xt_simple_dist)
shared_modules_dairv2xt_simple_dist_second['roi_head']['heads'][0]['sin_enc'] = True
shared_modules_dairv2xt_simple_dist_second['det1_head']['sin_enc'] = True
shared_modules_dairv2xt_simple_dist_second['det2_head']['sin_enc'] = True
#--------- Ablation 8: dilconv + expand_2d --------
shared_modules_dairv2xt_dilconv_exp2d = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_dilconv_exp2d['pts_backbone']['expand_2d'] = True
shared_modules_dairv2xt_dilconv_exp2d['pts_backbone']['expand_3d'] = False
shared_modules_dairv2xt_dilconv_exp2d['pts_backbone']['type'] = 'backbone3d.unet.UnetDilConv'
#--------- Ablation 10: sin cos box coder --------
shared_modules_dairv2xt_sin_cos = copy.deepcopy(shared_modules_dairv2xt_no_expand)
shared_modules_dairv2xt_sin_cos['roi_head']['heads'][0]['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_sin_cos['roi_head']['heads'][0]['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_dairv2xt_sin_cos['roi_head']['heads'][0]['reg_channels'] = ['box:6', 'dir:2', 'scr:0']
shared_modules_dairv2xt_sin_cos['det1_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_sin_cos['det1_head']['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_dairv2xt_sin_cos['det1_head']['reg_channels'] = ['box:6', 'dir:2', 'scr:0', 'vel:2']
shared_modules_dairv2xt_sin_cos['det2_head']['box_assigner']['box_coder']['type'] = 'CenterBoxCoderV2'
shared_modules_dairv2xt_sin_cos['det2_head']['box_assigner']['box_coder']['mode'] = 'sin_cos'
shared_modules_dairv2xt_sin_cos['det2_head']['reg_channels'] = ['box:6', 'dir:2', 'scr:0', 'vel:2']

#--------- Ablation 11: no expand 2d --------
shared_modules_dairv2xt_no_expand_2d = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_no_expand_2d['pts_backbone']['expand_2d'] = False
#--------- Ablation 12: second_head --------
from cosense3d.config.pycfg.template.det_anchor_sparse import get_det_anchor_sparse_cfg
shared_modules_dairv2xt_second = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_second['roi_head']['heads'][0] = get_det_anchor_sparse_cfg(
    voxel_size=voxel_size,
    point_cloud_range=dairv2xt.point_cloud_range,
    in_channels=256,
    get_roi_scores=True,
    box_coder_mode='simple_dist',
    target_assigner="BoxAnchorAssignerSecond",
    pos_threshold=0.3,
    neg_threshold=0.1
)
shared_modules_dairv2xt_second['det1_head'] = get_query_det_second_head_cfg(
            gather_keys=['temp_fusion_feat'],
            scatter_keys=['detection_local'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            # freeze=True,
            voxel_size=voxel_size,
            point_cloud_range=dairv2xt.point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        )
shared_modules_dairv2xt_second['det2_head'] = get_query_det_second_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            # freeze=True,
            voxel_size=voxel_size,
            point_cloud_range=dairv2xt.point_cloud_range,
            out_stride=out_stride,
            sparse=True,
        )

#--------- Ablation 14 : Attentive spatial fusion --------
shared_modules_dairv2xt_attentive = copy.deepcopy(shared_modules_dairv2xt)
shared_modules_dairv2xt_attentive['spatial_fusion']['type'] = 'fusion.spatial_query_fusion.SpatialAttentiveAlignment'


