from cosense3d.config import add_cfg_keys
from cosense3d.modules.plugin.target_assigners import BEVCenternessAssigner


@add_cfg_keys
def get_det_center_sparse_cfg(voxel_size, point_cloud_range,
                              in_channels=256, stride=2,
                              generate_roi_scr=False,
                              cls_assigner="BEVCenternessAssigner",
                              cls_loss="EDLLoss",
                              use_gaussian=False, sigma=1.0):
    if cls_assigner == "BEVCenternessAssigner":
        cls_assigner = dict(
                    type='target_assigners.BEVCenternessAssigner',
                    n_cls=1,
                    min_radius=1.0,
                    pos_neg_ratio=0,
                    max_mining_ratio=0,
                    use_gaussian=use_gaussian,
                    sigma=sigma
                )
    elif cls_assigner == "BEVBoxAssigner":
        cls_assigner = dict(
            type='target_assigners.BEVBoxAssigner',
            n_cls=1,
            pos_neg_ratio=0,
            max_mining_ratio=0,
        )
    else:
        raise NotImplementedError
    scr_activation = "relu" # default
    edl = True
    if cls_loss == "EDLLoss":
        cls_loss = dict(type='EDLLoss', activation='exp', annealing_step=20, n_cls=2, loss_weight=5.0)
        scr_activation = "exp"
        one_hot_encoding = True
    elif cls_loss == "FocalLoss":
        cls_loss = dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0)
        scr_activation = "sigmoid"
        edl = False
        one_hot_encoding = False

    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    return dict(
                type='heads.det_center_sparse.DetCenterSparse',
                data_info=data_info,
                generate_roi_scr=generate_roi_scr,
                input_channels=in_channels,
                shared_conv_channel=256,
                get_predictions=True,
                stride=stride,
                cls_head_cfg=dict(name='UnitedClsHead', one_hot_encoding=one_hot_encoding),
                reg_head_cfg=dict(name='UnitedRegHead', combine_channels=True, sigmoid_keys=['scr']),
                class_names_each_head=[['vehicle.car']],
                reg_channels=['box:6', 'dir:8', 'scr:4'],
                cls_assigner=cls_assigner,
                box_assigner=dict(
                    type='target_assigners.BoxCenterAssigner',
                    voxel_size=voxel_size,
                    lidar_range=point_cloud_range,
                    stride=stride,
                    detection_benchmark='Car',
                    class_names_each_head=[['vehicle.car']],
                    center_threshold=0.5,
                    box_coder=dict(type='CenterBoxCoder'),
                    activation=scr_activation,
                    edl=edl
                ),
                loss_cls=cls_loss,
                loss_box=dict(type='SmoothL1Loss', loss_weight=1.0),
            )