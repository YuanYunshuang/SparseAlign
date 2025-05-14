from cosense3d.config import add_cfg_keys

@add_cfg_keys
def get_det_anchor_sparse_cfg(voxel_size, point_cloud_range, in_channels=256, stride=2,
                             pos_threshold=0.6, neg_threshold=0.45, score_threshold=0.25,
                             get_roi_scores=False, box_coder_mode='simple_dist',
                             get_boxes_when_training=False, target_assigner="BoxAnchorAssigner"
                             ):
    return dict(
            type='heads.det_anchor_sparse.DetAnchorSparse',
            in_channels=in_channels,
            get_boxes_when_training=get_boxes_when_training,
            get_roi_scores=get_roi_scores,
            target_assigner=dict(
                type=f'target_assigners.{target_assigner}',
                box_size=[3.9, 1.6, 1.56],
                dirs=[0, 90],
                voxel_size=voxel_size,
                lidar_range=point_cloud_range,
                stride=stride,
                pos_threshold=pos_threshold,
                neg_threshold=neg_threshold,
                score_threshold=score_threshold,
                box_coder=dict(type='ResidualBoxCoder', mode=box_coder_mode)
            ),
            loss_cls = dict(type='FocalLoss', use_sigmoid=True,
                            gamma=2.0, alpha=0.25, loss_weight=0.25),
            loss_box = dict(type='SmoothL1Loss', loss_weight=1.0),
        )