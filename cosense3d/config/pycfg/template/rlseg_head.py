from cosense3d.config import add_cfg_keys

@add_cfg_keys
def get_roadline_head_cfg(data_info, stride, in_dim=256, range=50):
    res = data_info['voxel_size'][0] * stride
    return dict(
            type='heads.bev_roadline.BEVRoadLine',
            data_info=data_info,
            stride=stride,
            in_dim=in_dim,
            target_assigner=dict(type='target_assigners.RoadLineAssigner', res=res, range=range),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
        )


