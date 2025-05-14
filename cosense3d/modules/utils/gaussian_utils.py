from typing import List
import torch
from torch.distributions.multivariate_normal import _batch_mahalanobis
import torch_scatter
import numpy as np


def weighted_mahalanobis_dists(vars, dists, weights=None):
    """Compute the squared mahalanobis distances.

    :param vars: (N, 2), variances of Gaussian distribution.
    :param dists: (N, 2), distances to gaussian center at each axis.
    :param weights: weights to be applied to the output probability.
    :return: (N), squared mahalanobis
    """
    vars = vars.squeeze()
    if len(vars.shape) == 1:
        vars = torch.stack([vars, vars], dim=-1)
    covs = torch.diag_embed(vars.squeeze(), dim1=1)
    unbroadcasted_scale_tril = covs.unsqueeze(1)  # N 1 2 2

    # a.shape = (i, 1, n, n), b = (..., i, j, n),
    M = _batch_mahalanobis(unbroadcasted_scale_tril, dists)  # N M
    log_probs = -0.5 * M
    probs = log_probs.exp()  # N M 2
    if weights is not None:
        probs = probs * weights

    return probs


def mahalanobis_dists_2d(sigmas, dists):
    """Compute the squared mahalanobis distances.

    :param sigmas: (N, 2), standard deviation of Gaussian distribution
    :param dists: (N, 2), distances to gaussian center
    :return: (N), squared mahalanobis
    """
    vars = sigmas ** 2
    covs = torch.diag_embed(vars, dim1=1)
    unbroadcasted_scale_tril = covs.unsqueeze(1)  # 1 1 2 2
    M = -0.5 * _batch_mahalanobis(unbroadcasted_scale_tril, dists.unsqueeze(0))  # N M
    return M


def center_to_img_coor(center_in, lidar_range, pixel_sz):
    x, y = center_in[:, 0], center_in[:, 1]
    coord_x = (x - lidar_range[0]) / pixel_sz
    coord_y = (y - lidar_range[1]) / pixel_sz
    map_sz_x = (lidar_range[3] - lidar_range[0]) / pixel_sz
    map_sz_y = (lidar_range[4] - lidar_range[1]) / pixel_sz
    # clamp to fit image size: 1e-6 does not work for center.int()
    coord_x = torch.clamp(coord_x, min=0, max=map_sz_x - 0.5)
    coord_y = torch.clamp(coord_y, min=0, max=map_sz_y - 0.5)
    center_out = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
    return center_out


def cornernet_gaussian_radius(height, width, min_overlap=0.5):
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian_radius(box_dims, pixel_sz, overlap, min_radius=2):
    dx, dy = box_dims[:, 0] / pixel_sz[0], box_dims[:, 1] / pixel_sz[1]

    radius = cornernet_gaussian_radius(dx, dy, min_overlap=overlap)
    radius = torch.clamp_min(radius.int(), min=min_radius)

    return radius


def gaussian_2d(shape: List[int], sigma: float=1.0) -> np.ndarray:
    """Generate gaussian map.

    :param shape: Shape of the map.
    :param sigma: Sigma to generate gaussian map.
            Defaults to 1.
    :return: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_map(boxes, lidar_range, pixel_sz, batch_size, radius=None, sigma=1, min_radius=2):
    size_x = int((lidar_range[3] - lidar_range[0]) // pixel_sz[0])
    size_y = int((lidar_range[4] - lidar_range[1]) // pixel_sz[1])
    if boxes.shape[0] == 0:
        return torch.zeros(batch_size, size_x, size_y, device=boxes.device)
    if radius is None:
        radius = torch.ones_like(boxes[:, 0]) * 2
    radius_max = radius.max()
    center = center_to_img_coor(boxes[:, 1:3], lidar_range, pixel_sz)
    ctridx = center.int()

    # sample points for each center point
    steps = radius_max * 2 + 1
    x = torch.linspace(- radius_max, radius_max, steps)
    offsets = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).to(center.device)
    offsets = offsets[torch.norm(offsets, dim=-1) <= radius_max]
    samples = ctridx.unsqueeze(1) + offsets.view(1, -1, 2)
    ind = torch.tile(boxes[:, 0].unsqueeze(1), (1, samples.shape[1])).unsqueeze(-1)
    samples = torch.cat([ind, samples], dim=-1)
    ctr_idx_of_sam = torch.arange(len(center)).unsqueeze(1).tile(1, samples.shape[1])

    mask = (samples[..., 1] >= 0) & (samples[..., 1] < size_x) & \
           (samples[..., 2] >= 0) & (samples[..., 2] < size_y)


    new_center = center[ctr_idx_of_sam[mask]]
    new_vars = 1 / min_radius * radius[ctr_idx_of_sam[mask]].float()
    new_samples = samples[mask]
    dists_sam2ctr = new_samples[:, 1:].float() - new_center

    probs = weighted_mahalanobis_dists(
        new_vars,
        dists_sam2ctr.unsqueeze(1),
    ).squeeze()

    # probs = probs / (2 * sigma * sigma)
    probs[probs < torch.finfo(probs.dtype).eps * probs.max()] = 0

    indices = new_samples[:, 0] * size_y * size_x + \
              new_samples[:, 1] * size_x + new_samples[:, 2]

    center_map = torch.zeros(batch_size * size_x * size_y, device=center.device)
    torch_scatter.scatter(probs, indices.long(), dim=0, out=center_map, reduce='max')
    center_map = center_map.view(batch_size, size_x, size_y)

    # import matplotlib.pyplot as plt
    # plt.imshow(center_map[0].cpu().numpy())
    # plt.show()
    # plt.close()

    return center_map