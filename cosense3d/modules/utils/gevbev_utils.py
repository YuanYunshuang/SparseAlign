import torch
from torch.distributions.multivariate_normal import _batch_mahalanobis

from cosense3d.modules.utils.me_utils import metric2indices
from cosense3d.modules.utils.common import meshgrid, pad_l


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


def draw_sample_evis(ctr_pts: dict, samples: torch.Tensor, tag: str,
                     res: float, distr_r: float, lr: list,
                     batch_size: int, var0: float)->torch.Tensor:
    """
    Given center points and its regression results, generate evidences for new samples.

    Parameters
    ----------
    ctr_pts: points in BEV feature map, including its
        - (key='ctr') metric center coordinates ,
        - (key='coor') index coordinates  and
        - (key='reg') regression values for centers, including the EDL evidence for each class and 2D stds for each class.
    reg:
    samples: sparse target points that are sampled in the continuous BEV space.
    tag: tag for regression key.
    res: resolution of the center points.
    distr_r: maximum radius of the Gaussian distribution over which to draw samples.
    lr: lidar range.
    batch_size: batch size.
    var0: base variance, to be added to the regressed variances.

    Returns
    -------
    Evidences for the given samples.
    """
    if len(samples) == 0:
        return torch.empty((0, 2), device=ctr_pts[f'reg_{tag}'].device)
    mask = (ctr_pts['ctr'].abs() < lr[3]).all(1)
    if mask.sum() == 0:
        return torch.zeros_like(samples[:, :2])
    reg = ctr_pts[f'reg_{tag}'][mask].relu()
    ctr = ctr_pts['ctr'][mask]
    coor = ctr_pts['coor'][mask]
    assert reg.shape[1] == 6
    reg_evi = reg[:, :2]
    reg_var = reg[:, 2:].view(-1, 2, 2)

    # create index map for center points
    grid_size = (round((lr[3] - lr[0]) / res), round((lr[4] - lr[1]) / res))
    centers_map = torch.ones((batch_size, grid_size[0], grid_size[1]),
                              device=reg.device).long() - 1
    ctridx = coor.clone().T
    ctridx[1] -= round(lr[0] / res)
    ctridx[2] -= round(lr[1] / res)
    ctridx = ctridx.long()
    centers_map[ctridx[0], ctridx[1], ctridx[2]] = torch.arange(ctridx.shape[1],
                                                                device=ctridx.device)

    # get neighboring center indices for sample points
    steps = int(distr_r / res)
    offset = meshgrid(-steps, steps, 2, n_steps=steps * 2 + 1).to(samples.device) # s s 2
    samidx = metric2indices(samples[:, :3], res).view(-1, 1, 3) \
             + pad_l(offset).view(1, -1, 3)  # n s*s 3
    samidx = samidx.view(-1, 3).T  # 3 n*s*s
    samidx[1] = (samidx[1] - (lr[0] / res))
    samidx[2] = (samidx[2] - (lr[1] / res))
    mask1 = (samidx[1] >= 0) & (samidx[1] < grid_size[0]) & \
            (samidx[2] >= 0) & (samidx[2] < grid_size[1])
    inds = samidx[:, mask1].long()
    ctr_idx_of_sam = centers_map[inds[0], inds[1], inds[2]]
    mask2 = ctr_idx_of_sam >= 0
    ctr_idx_of_sam = ctr_idx_of_sam[mask2]
    ns = offset.shape[0]**2
    new_samples = torch.tile(samples[:, :3].unsqueeze(1),
                             (1, ns, 1)).view(-1, 3)  # n*s*s 3
    new_centers = ctr[ctr_idx_of_sam]
    dists_sam2ctr = new_samples[mask1][mask2][:, 1:] - new_centers[:, 1:]

    probs_weighted = weighted_mahalanobis_dists(
        reg_evi[ctr_idx_of_sam],
        reg_var[ctr_idx_of_sam],
        dists_sam2ctr.unsqueeze(1),
        var0=[var0] * 2
    ).squeeze()

    sample_evis = torch.zeros_like(samidx[:2].T)
    mask = mask1.clone()
    mask[mask1] = mask2
    sample_evis[mask] = probs_weighted
    sample_evis = sample_evis.view(-1, ns, 2).sum(dim=1)

    if sample_evis.isnan().any():
        print('d')

    return sample_evis