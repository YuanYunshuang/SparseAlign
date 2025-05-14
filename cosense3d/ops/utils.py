import copy
import torch
import numpy as np
import cuda_ops
from cosense3d.utils.misc import check_numpy_to_torch


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


@torch.no_grad()
def decompose_batch_indices(tensor, batch_size, batch_indices):
    if batch_indices is None:
        batch_indices = range(batch_size)
    points_decomposed = [tensor[tensor[:, 0] == b] for b in batch_indices]
    decomposed_tensor = torch.cat(points_decomposed, dim=0)
    cnts = [len(pts) for pts in points_decomposed]
    out_tensor = torch.zeros((batch_size, max(cnts), tensor.shape[-1] - 1),
                             dtype=tensor.dtype, device=tensor.device)
    for b, (c, points) in enumerate(zip(cnts, points_decomposed)):
        out_tensor[b, :c, :] = points_decomposed[b][:, 1:]
    return decomposed_tensor, out_tensor, cnts


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = check_numpy_to_torch(points)
    boxes, is_numpy = check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    cuda_ops.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices


@torch.no_grad()
def points_in_boxes_gpu(points, boxes, batch_size=None, batch_indices=None):
    """
    :param points: (B, M, 3) or (M, 4)
    :param boxes: (B, T, 7) or (T, 8), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    src_idx = points[:, 0]
    batch_flag = False
    if len(points.shape)==2:
        assert batch_size is not None
        assert boxes[:, 0].max() < batch_size and points[:, 0].max() < batch_size, \
            f"Box shape: {', '.join([str(s) for s in boxes.shape])}"
        assert boxes.shape[1] == 8 and points.shape[1] == 4
        batch_flag = True
        n_box = len(boxes)
        _, points, point_cnts = decompose_batch_indices(points, batch_size, batch_indices)
        boxes_decomposed, boxes, box_cnts = decompose_batch_indices(boxes, batch_size, batch_indices)
    assert boxes.shape[0] == points.shape[0], \
        f"boxes and point batch size does not match! boxes ({boxes.shape[0]}), points ({points.shape[0]})"
    assert boxes.shape[2] == 7 and points.shape[2] == 3

    batch_size, num_points, _ = points.shape

    box_idxs_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    cuda_ops.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idxs_of_pts)

    if batch_flag:
        box_idxs_composed = torch.zeros(len(src_idx), dtype=torch.int,
                                        device=points.device).fill_(-1)
        cnt_p = 0
        cnt_b = 0
        for b, (cp, cb) in enumerate(zip(point_cnts, box_cnts)):
            indices = box_idxs_of_pts[b, :cp]
            if cb == 0:
                # zero points are assigned to padded zeros boxes --> remove them
                indices = -1
            else:
                indices[indices >= 0] += cnt_b
            box_idxs_composed[src_idx==b] = indices
            cnt_p += cp
            cnt_b += cb
        return boxes_decomposed, box_idxs_composed.long()
    return box_idxs_of_pts.long()


@torch.no_grad()
def points_in_boxes_gpu_2d(points, boxes, batch_size=None, batch_indices=None):
    cur_boxes = copy.deepcopy(boxes)
    cur_boxes[:, 3] = 0
    bev_pts = copy.deepcopy(points)
    bev_pts[:, 3] = 0
    _, box_idx_of_pts = points_in_boxes_gpu(bev_pts, cur_boxes,
                                            batch_size=batch_size,
                                            batch_indices=batch_indices)

    return box_idx_of_pts