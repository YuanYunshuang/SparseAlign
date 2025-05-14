import copy
from collections import OrderedDict
from cosense3d.config.pycfg.base import opv2v, hooks
from cosense3d.config.pycfg.template.minkunet import get_minkunet_cfg
from cosense3d.config.pycfg.template.bev_semseg_head import get_bev_semseg_head_cfg
from cosense3d.config.pycfg.template.det_center_sparse import get_det_center_sparse_cfg



def get_shared_modules(point_cloud_range, in_dim=4, version='gevbev', det=True, vs=0.2, out_stride=2,
                       unet_enc_dim=32):
    """
    gather_keys: 
        keys to gather data from cavs, key order is important, should match the forward input arguments order.
    scatter_keys: 
        1st key in the list is used as the key for scattering and storing module output data to cav.
    """
    voxel_size = [vs, vs, vs]
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    strides = [out_stride]
    dconvs = {f"p{out_stride}": dict(kernels=[5, 5, 3], in_dim=384, out_dim=256),}
    if 'gevbev' == version:
        semseg_head_type = 'heads.bev_semseg.GevSemsegHead'
        tgt_assigner_type = 'target_assigners.ContiBEVAssigner'
    else:
        tgt_assigner_type = 'target_assigners.DiscreteBEVAssigner'
        semseg_head_type = 'heads.bev_semseg.EviSemsegHead'
    if det:
        strides = [out_stride, out_stride * 2]
        dconvs = {
            f"p{out_stride}": dict(kernels=[5, 5, 3], in_dim=384, out_dim=256),
            f"p{out_stride * 2}": dict(kernels=[3, 3, 3], in_dim=384, out_dim=256),
        }

    cfg = OrderedDict(
        pts_backbone=get_minkunet_cfg(
            gather_keys=['points'],
            scatter_keys=['bev_feat'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            height_compression=strides,
            compression_kernel_size_xy=3,
            cache_strides=strides,
            in_dim=in_dim,
            enc_dim=unet_enc_dim,
        ),

        semseg_head_local=get_bev_semseg_head_cfg(
            gather_keys=['bev_feat'],
            scatter_keys=['bev_semseg_local'],
            gt_keys=['bev_tgt_pts', 'local_bboxes_3d'],
            semseg_head_type=semseg_head_type,
            data_info=data_info,
            stride=out_stride,
            in_dim=384,
            tgt_assigner_type=tgt_assigner_type,
        ),

        spatial_fusion=dict(
            type='fusion.naive_fusion.NaiveFusion',
            gather_keys=['bev_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            stride=strides
        ),

        fusion_neck=dict(
            type='necks.dilation_spconv.DilationSpconv',
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['spatial_fusion_feat'],
            data_info=data_info,
            d=2,
            convs=dconvs,
            n_layers=6,
        ),

        semseg_head=get_bev_semseg_head_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['bev_semseg'],
            gt_keys=['global_bev_tgt_pts', 'global_bboxes_3d'],
            semseg_head_type=semseg_head_type,
            data_info=data_info,
            stride=out_stride,
            in_dim=256,
            tgt_assigner_type=tgt_assigner_type,
        ),

        det_head=get_det_center_sparse_cfg(
            gather_keys=['spatial_fusion_feat'],
            scatter_keys=['detection'],
            gt_keys=['global_bboxes_3d', 'global_labels_3d'],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            in_channels=256,
            stride=out_stride * 2,
            cls_loss = "FocalLoss",
        )
    )

    if not det:
        cfg.pop('det_head')
    return cfg

######################################################
#                     OPV2V
######################################################
test_hooks_opv2v_bev = (hooks.get_test_bev_semseg_hooks(opv2v.point_cloud_range_bev_test))
test_hooks_opv2v_bev_det = (hooks.get_test_bev_semseg_hooks(opv2v.point_cloud_range_bev_test)
                            + hooks.get_test_nms_eval_hooks(opv2v.point_cloud_range_test))
plots_opv2v = [hooks.get_detection_plot(opv2v.point_cloud_range_test)]
shared_modules_gevbev_opv2v = get_shared_modules(opv2v.point_cloud_range, version='gevbev', det=False)
shared_modules_gevbev_with_det_opv2v = get_shared_modules(opv2v.point_cloud_range, version='gevbev', det=True)
shared_modules_evibev_opv2v = get_shared_modules(opv2v.point_cloud_range, version='evibev')

#--------- Ablation 1 : Attention fusion -------------
shared_modules_gevbev_with_det_opv2v_attn = get_shared_modules(opv2v.point_cloud_range, version='gevbev', det=True)
shared_modules_gevbev_with_det_opv2v_attn['spatial_fusion'] = dict(
            type='fusion.attn_fusion.SparseAttentionFusion',
            gather_keys=['bev_feat', 'received_response'],
            scatter_keys=['spatial_fusion_feat'],
            stride=[2, 4],
            in_channels=348,
        )

#--------- Ablation 2 : in_dim=7 -------------
shared_modules_gevbev_opv2v_naive_in7 = get_shared_modules(opv2v.point_cloud_range_bev, version='gevbev', in_dim=7)

#--------- Ablation 3 : voxel_size = 0.4m -------------
shared_modules_gevbev_opv2v_naive_in3_v4 = get_shared_modules(opv2v.point_cloud_range_bev, version='gevbev', vs=0.4, out_stride=1, unet_enc_dim=64)