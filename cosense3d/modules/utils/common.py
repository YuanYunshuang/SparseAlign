from importlib import import_module

import torch
from torch import nn
import numpy as np

from torch.distributions.multivariate_normal import _batch_mahalanobis
from cosense3d.modules.utils.me_utils import metric2indices

pi = 3.141592653


def clip_sigmoid(x: torch.Tensor, eps: float=1e-4) -> torch.Tensor:
    """Sigmoid function for input feature.

    :param x: Input feature map with the shape of [B, N, H, W].
    :param eps: Lower bound of the range to be clamped to.
            Defaults to 1e-4.
    :return: Feature map after sigmoid.
    """
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


def cat_name_str(module_name):
    """
    :param module_name: str, format in xxx_yyy_zzz
    :returns: class_name: str, format in XxxYyyZzz
    """
    cls_name = ''
    for word in module_name.split('_'):
        cls_name += word[:1].upper() + word[1:]
    return cls_name


def instantiate(module_name, cls_name=None, module_cfg=None, **kwargs):
    package = import_module(f"cosense3d.model.{module_name}")
    cls_name = cat_name_str(module_name) if cls_name is None else cls_name
    obj_cls = getattr(package, cls_name)
    if module_cfg is None:
        obj_inst = obj_cls(**kwargs)
    else:
        obj_inst = obj_cls(module_cfg)
    return obj_inst


def bias_init_with_prob(prior_prob: float) -> float:
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape

        view_shape = [1 for _ in range(len(feat_shape))]
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)

        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    :param x: (Tensor) The tensor to do the
            inverse.
    :param eps: (float) EPS avoid numerical
            overflow. Defaults 1e-5.
    :returns: Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


def get_conv2d_layers(conv_name, in_channels, out_channels, n_layers, kernel_size, stride,
                    padding, relu_last=True, sequential=True, **kwargs):
    """
    Build convolutional layers. kernel_size, stride and padding should be a list with the
    lengths that match n_layers
    """
    seq = []
    if 'bias' in kwargs:
        bias = kwargs.pop('bias')
    else:
        bias = False
    for i in range(n_layers):
        seq.extend([getattr(nn, conv_name)(
            in_channels, out_channels, kernel_size[i], stride=stride[i],
            padding=padding[i], bias=bias, **{k: v[i] for k, v in kwargs.items()}
        ), nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)])
        if i < n_layers - 1 or relu_last:
            seq.append(nn.ReLU())
        in_channels = out_channels
    if sequential:
        return nn.Sequential(*seq)
    else:
        return seq


def get_norm_layer(channels, norm):
    if norm == 'LN':
        norm_layer = nn.LayerNorm(channels)
    elif norm == 'BN':
        norm_layer = nn.BatchNorm1d(channels)
    else:
        raise NotImplementedError
    return norm_layer


def linear_last(in_channels, mid_channels, out_channels, bias=False, norm='BN'):
    return nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=bias),
            get_norm_layer(mid_channels, norm),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels)
        )


def linear_layers(in_out, activations=None, norm='BN'):
    if activations is None:
        activations = ['ReLU'] * (len(in_out) - 1)
    elif isinstance(activations, str):
        activations = [activations] * (len(in_out) - 1)
    else:
        assert len(activations) == (len(in_out) - 1)
    layers = []
    for i in range(len(in_out) - 1):
        layers.append(nn.Linear(in_out[i], in_out[i+1], bias=False))
        layers.append(get_norm_layer(in_out[i+1], norm))
        layers.append(getattr(nn, activations[i])())
    return nn.Sequential(*layers)


def meshgrid(xmin, xmax, ymin=None, ymax=None, dim=2, n_steps=None, step=None):
    assert dim <= 3, f'dim <= 3, but dim={dim} is given.'
    if ymin is not None and ymax is not None:
        assert dim == 2
        if n_steps is not None:
            x = torch.linspace(xmin, xmax, n_steps)
            y = torch.linspace(ymin, ymax, n_steps)
        elif step is not None:
            x = torch.arange(xmin, xmax, step)
            y = torch.arange(ymin, ymax, step)
        else:
            raise NotImplementedError
        xs = (x, y)
    else:
        if n_steps is not None:
            x = torch.linspace(xmin, xmax, n_steps)
            if ymin is not None and ymax is not None:
                y = torch.linspace(ymin, ymax, n_steps)
        elif step is not None:
            x = torch.arange(xmin, xmax, step)
        else:
            raise NotImplementedError
        xs = (x, ) * dim
    indexing = 'ijk'
    indexing = indexing[:dim]
    coor = torch.stack(
        torch.meshgrid(*xs, indexing=indexing),
        dim=-1
    )
    return coor


def meshgrid_cross(xmins, xmaxs, n_steps=None, steps=None):
    if n_steps is not None:
        assert len(xmins) == len(n_steps)
        xs = [torch.linspace(xmin, xmax + 1, nstp) for xmin, xmax, nstp \
             in zip(xmins, xmaxs, n_steps)]
    elif steps is not None:
        xs = [torch.arange(xmin, xmax + 1, stp) for xmin, xmax, stp \
             in zip(xmins, xmaxs, steps)]
    else:
        raise NotImplementedError
    dim = len(xs)
    indexing = 'ijk'
    indexing = indexing[:dim]
    coor = torch.stack(
        torch.meshgrid(*xs, indexing=indexing),
        dim=-1
    )
    return coor

def pad_r(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor, tensor_pad], dim=-1)


def pad_l(tensor, value=0):
    tensor_pad = torch.ones_like(tensor[..., :1]) * value
    return torch.cat([tensor_pad, tensor], dim=-1)


def cat_coor_with_idx(tensor_list):
    out = []
    for i, t in enumerate(tensor_list):
        out.append(pad_l(t, i))
    return torch.cat(out, dim=0)


def fuse_batch_indices(coords, num_cav):
    """
    Fusing voxels of CAVs from the same frame
    :param stensor: ME sparse tensor
    :param num_cav: list of number of CAVs for each frame
    :return: fused coordinates and features of stensor
    """

    for i, c in enumerate(num_cav):
        idx_start = sum(num_cav[:i])
        mask = torch.logical_and(
            coords[:, 0] >= idx_start,
            coords[:, 0] < idx_start + c
        )
        coords[mask, 0] = i

    return coords


def weighted_mahalanobis_dists(reg_evi, reg_var, dists, var0):
    log_probs_list = []
    for i in range(reg_evi.shape[1]):
        vars = reg_var[:, i, :] + var0[i]
        covs = torch.diag_embed(vars.squeeze(), dim1=1)
        unbroadcasted_scale_tril = covs.unsqueeze(1)  # N 1 2 2

        # a.shape = (i, 1, n, n), b = (..., i, j, n),
        M = _batch_mahalanobis(unbroadcasted_scale_tril, dists)  # N M
        log_probs = -0.5 * M
        log_probs_list.append(log_probs)

    log_probs = torch.stack(log_probs_list, dim=-1)
    probs = log_probs.exp()  # N M 2
    cls_evi = reg_evi.view(-1, 1, 2)  # N 1 2
    probs_weighted = probs * cls_evi

    return probs_weighted


def draw_sample_prob(centers, reg, samples, res, distr_r, det_r, batch_size, var0):
    # from utils.vislib import draw_points_boxes_plt
    # vis_ctrs = centers[centers[:, 0]==0, 1:].cpu().numpy()
    # vis_sams = samples[samples[:, 0]==0, 1:].cpu().numpy()
    #
    # ax = draw_points_boxes_plt(50, vis_ctrs, points_c='det_r', return_ax=True)
    # draw_points_boxes_plt(50, vis_sams, points_c='b', ax=ax)
    reg_evi = reg[:, :2]
    reg_var = reg[:, 2:].view(-1, 2, 2)

    grid_size = int(det_r / res) * 2
    centers_map = torch.ones((batch_size, grid_size, grid_size),
                              device=reg.device).long() * -1
    ctridx = metric2indices(centers, res).T
    ctridx[1:] += int(grid_size / 2)
    centers_map[ctridx[0], ctridx[1], ctridx[2]] = torch.arange(ctridx.shape[1],
                                                                device=ctridx.device)

    steps = int(distr_r / res)
    offset = meshgrid(-steps, steps, 2, n_steps=steps * 2 + 1).to(samples.device) # s s 2
    samidx = metric2indices(samples, res).view(-1, 1, 3) \
             + pad_l(offset).view(1, -1, 3)  # n s*s 3
    samidx = samidx.view(-1, 3).T  # 3 n*s*s
    samidx[1:] = (samidx[1:] + (det_r / res))
    mask1 = torch.logical_and((samidx[1:] >= 0).all(dim=0),
                             (samidx[1:] < (det_r / res * 2)).all(dim=0))

    inds = samidx[:, mask1].long()
    ctr_idx_of_sam = centers_map[inds[0], inds[1], inds[2]]
    mask2 = ctr_idx_of_sam >= 0
    ctr_idx_of_sam = ctr_idx_of_sam[mask2]
    ns = offset.shape[0]**2
    new_samples = torch.tile(samples.unsqueeze(1),
                             (1, ns, 1)).view(-1, 3)  # n*s*s 3
    new_centers = centers[ctr_idx_of_sam]
    dists_sam2ctr = new_samples[mask1][mask2][:, 1:] - new_centers[:, 1:]

    probs_weighted = weighted_mahalanobis_dists(
        reg_evi[ctr_idx_of_sam],
        reg_var[ctr_idx_of_sam],
        dists_sam2ctr.unsqueeze(1),
        var0=var0
    ).squeeze()

    sample_evis = torch.zeros_like(samidx[:2].T)
    mask = mask1.clone()
    mask[mask1] = mask2
    sample_evis[mask] = probs_weighted
    sample_evis = sample_evis.view(-1, ns, 2).sum(dim=1)

    return sample_evis


def get_voxel_centers(voxel_coords,
                      downsample_times,
                      voxel_size,
                      point_cloud_range):
    """Get centers of spconv voxels.

    :param voxel_coords: (N, 3)
    :param downsample_times:
    :param voxel_size:
    :param point_cloud_range:
    :return:
    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


