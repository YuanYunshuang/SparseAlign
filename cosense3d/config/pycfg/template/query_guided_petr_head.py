

def get_query_guided_petr_head_cfg(gather_keys, scatter_keys, gt_keys,
                                   voxel_size, point_cloud_range, out_stride,
                                   embed_dims=256, sparse=False, freeze=False,
                                   pred_while_training=False):
    return dict(
            type='heads.query_guided_petr_head.QueryGuidedPETRHead',
            freeze=freeze,
            gather_keys=gather_keys,
            scatter_keys=scatter_keys,
            gt_keys=gt_keys,
            sparse=sparse,
            pred_while_training=pred_while_training,
            embed_dims=embed_dims,
            num_reg_fcs=1,
            num_pred=1,
            pc_range=point_cloud_range,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            num_classes=1,
            reg_channels=['box:6', 'dir:8', 'scr:4', 'vel:2'],
            cls_assigner=dict(
                type='target_assigners.BEVBoxAssigner',
                n_cls=1,
                # min_radius=1.0,
                pos_neg_ratio=0,
                mining_thr=0,
            ),
            box_assigner=dict(
                type='target_assigners.BoxCenterAssigner',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                stride=out_stride,
                detection_benchmark='Car',
                class_names_each_head=[['vehicle.car']],
                center_threshold=0.5,
                box_coder=dict(type='CenterBoxCoder', with_velo=True),
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                          gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
        )


def get_query_guided_pred_head_cfg(gather_keys, scatter_keys, gt_keys,
        voxel_size, point_cloud_range, out_stride, embed_dims=256, sparse=False):
    return dict(
            type='heads.query_guided_petr_head.QueryGuidedPredHead',
            gather_keys=gather_keys,
            scatter_keys=scatter_keys,
            gt_keys=gt_keys,
            sparse=sparse,
            embed_dims=embed_dims,
            num_reg_fcs=1,
            num_pred=1,
            pc_range=point_cloud_range,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            num_classes=1,
            reg_channels=['box:6', 'dir:8', 'scr:4', 'vel:2'],
            cls_assigner=dict(
                type='target_assigners.BEVCenternessAssigner',
                n_cls=1,
                min_radius=1.0,
                pos_neg_ratio=0,
                mining_thr=0,
            ),
            box_assigner=dict(
                type='target_assigners.BoxCenterAssigner',
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                stride=out_stride,
                detection_benchmark='Car',
                class_names_each_head=[['vehicle.car']],
                center_threshold=0.5,
                box_coder=dict(type='BoxPredCoder', with_velo=True),
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                          gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
        )