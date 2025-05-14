from cosense3d.config import add_cfg_keys

def get_bev_head_cfg(data_info, out_stride, in_dim=256, n_cls=2):
    return dict(
        type='heads.bev.BEV',
        data_info=data_info,
        stride=out_stride,
        in_dim=in_dim,
        target_assigner=dict(type='target_assigners.BEVPointAssigner'),
        loss_cls=dict(type='EDLLoss', activation='relu', annealing_step=50,
                      n_cls=n_cls, loss_weight=1.0),
    )


@add_cfg_keys
def get_bev_multi_resolution_head_cfg(data_info, in_dim=256, n_cls=1):
    cfg = dict(
        type='heads.bev.BEVMultiResolution',
        data_info=data_info,
        strides=[2, 8],
        strides_for_loss=[2],
        down_sample_tgt=False,
        in_dim=in_dim,
        num_cls=n_cls,
        target_assigner=dict(type='target_assigners.BEVPointAssigner', down_sample=False),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, bg_idx=0,
                      gamma=2.0, alpha=0.25, loss_weight=2.0),
    )
    return cfg