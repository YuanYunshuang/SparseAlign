import numpy as np
import torch
from typing import Union

from shapely.geometry import Polygon

from cosense3d.utils.misc import check_numpy_to_torch
from cosense3d.ops.utils import points_in_boxes_cpu
from cosense3d.utils.pclib import rotate_points_batch, rotation_mat2euler_torch



def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period


def decode_boxes(reg, points, lwh_mean):
    assert len(reg)==len(points)
    if not isinstance(lwh_mean, torch.Tensor):
        lwh_mean = torch.Tensor(lwh_mean).view(1, 3)
    points = points.to(reg.device)
    lwh_mean = lwh_mean.to(reg.device)

    diagonal = torch.norm(lwh_mean[0, :2])
    # encode with diagonal length
    xy = reg[:, :2] * diagonal + points[:, :2]
    z = reg[:, 2:3] * lwh_mean[0, 2] + points[:, 2:3]
    lwh = reg[:, 3:6].exp() * lwh_mean
    r = torch.atan2(reg[:, 6:7], reg[:, 7:])

    return torch.cat([xy, z, lwh, r], dim=-1)


def boxes_to_corners_2d(boxes_np):
    """
    Convert boxes to 4 corners in xy plane
    :param boxes_np: np.ndarray [N, 7], cols - (x,y,z,dx,dy,dz,det_r)
    :return: corners: np.ndarray [N, 4, 2], corner order is
    back left, front left, front back, back left
    """
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = - dx / 2
    y1 = - dy / 2
    x2 = + dx / 2
    y2 = + dy / 2
    theta = boxes_np[:, 6:7]
    # bl, fl, fr, br
    corners = np.array([[x1, y2],[x2,y2], [x2,y1], [x1, y1]]).transpose(2, 0, 1)
    new_x = corners[:, :, 0] * np.cos(theta) + \
            corners[:, :, 1] * -np.sin(theta) + x[:, None]
    new_y = corners[:, :, 0] * np.sin(theta) + \
            corners[:, :, 1] * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)

    return corners


def boxes_to_corners_3d(boxes3d: Union[np.ndarray, torch.Tensor],
                        order: str='lwh'
                        ) -> Union[np.ndarray, torch.Tensor]:
    r"""
        4 -------- 5             ^ z
       /|         /|             |
      7 -------- 6 .             |
      | |        | |             | . x
      . 0 -------- 1             |/
      |/         |/              +-------> y
      3 -------- 2

    :param boxes3d: (N, 7 + (2: optional)) [x, y, z, dx, dy, dz, yaw]
    or [x, y, z, dx, dy, dz, roll, pitch, yaw], (x, y, z) is the box center.
    :param order: 'lwh' or 'hwl'.
    :return: (N, 8, 3), the 8 corners of the bounding box.
    """
    assert isinstance(boxes3d, np.ndarray) \
           or isinstance(boxes3d, torch.Tensor),\
    "input boxes should be numpy array or torch tensor."
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    if order == 'hwl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 4, 3]]
    elif order == 'lwh':
        pass

    template = boxes3d.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    if boxes3d[:, 6:].shape[1] == 1:
        rot_order = 'z'
    elif boxes3d[:, 6:].shape[1] == 3:
        rot_order = 'xyz'
    else:
        raise IOError("box input shape should be (N, 7) for (N, 9).")

    corners3d = rotate_points_batch(corners3d.view(-1, 8, 3),
                                    boxes3d[:, 6:], order=rot_order).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def corners_to_boxes_3d(corners: Union[np.ndarray, torch.Tensor],
                        mode: int=9
                        ) -> Union[np.ndarray, torch.Tensor]:
    r"""
        4 -------- 5             ^ z
       /|         /|             |
      7 -------- 6 .             |
      | |        | |             | . x
      . 0 -------- 1             |/
      |/         |/              +-------> y
      3 -------- 2

    :param corners: (N, 8, 3)
    :param mode: 9 | 7
    :return: boxes, (N, 9 | 7)
    """
    corners, is_numpy = check_numpy_to_torch(corners)
    xyz = corners.mean(axis=1)
    corners_reduced = corners - xyz.reshape(-1, 1, 3)
    diff_x = corners[:, [0, 1, 5, 4], :] - corners[:, [3, 2, 6, 7], :]
    diff_y = corners[:, [1, 5, 6, 2], :] - corners[:, [0, 4, 7, 3], :]
    diff_z = corners[:, [4, 5, 6, 7], :] - corners[:, [0, 1, 2, 3], :]
    l = torch.norm(diff_x, dim=2).mean(dim=1).reshape(-1, 1)
    w = torch.norm(diff_y, dim=2).mean(dim=1).reshape(-1, 1)
    h = torch.norm(diff_z, dim=2).mean(dim=1).reshape(-1, 1)

    template = corners.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )).reshape(1, 8, 3) * torch.cat([l, w, h], dim=1)[:, None, :] / 2
    R, _ = find_rigid_alignment(template, corners_reduced)
    euler = rotation_mat2euler_torch(R)
    # yaw = torch.arctan2(dir_x[:, 1], dir_x[:, 0]).reshape(-1, 1)
    if mode == 9:
        boxes = torch.cat([xyz, l, w, h, euler], dim=1)
    elif mode == 7:
        boxes = torch.cat([xyz, l, w, h, euler[:, -1:]], dim=1)
    else:
        raise NotImplementedError
    return boxes.numpy() if is_numpy else boxes


def boxes3d_to_standup_bboxes(boxes):
    """
    :param boxes: Tensor(N, 7)
    :return: Tenosr(N, 4): [x_min, y_min, x_max, y_max)
    """
    corners = boxes_to_corners_3d(boxes)
    standup_boxes = torch.zeros_like(boxes[:, :4])
    standup_boxes[:, :2] = corners[..., :2].min(dim=1)[0]
    standup_boxes[:, 2:] = corners[..., :2].max(dim=1)[0]
    return standup_boxes


def find_rigid_alignment(A, B):
    """Find rotation and translation from A to B.
    Parameters

    :param A: (B, N, 3)
    :param B: (B, N, 3)
    :return:
    """
    A_mean = A.mean(dim=1, keepdim=True)
    B_mean = B.mean(dim=1, keepdim=True)
    A_c = A - A_mean
    B_c = B - B_mean
    # Covariance matrix
    H = torch.bmm(A_c.permute(0, 2, 1), B_c)  # (B, 3, N) @ (B, N, 3) = (B, 3, 3)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = torch.bmm(V, U.permute(0, 2, 1))
    # Translation vector
    t = B_mean[:, None, :] - torch.bmm(R, A_mean.permute(0, 2, 1)).permute(0, 2, 1)
    return R, t


def mask_boxes_outside_range_numpy(boxes: np.ndarray,
                                   limit_range: list,
                                   order: str,
                                   min_num_corners: int=2) -> np.ndarray:
    """

    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param limit_range: [minx, miny, minz, maxx, maxy, maxz]
    :param order: 'lwh' or 'hwl'
    :param min_num_corners: The required minimum number of corners to be considered as in range.
    :return: The filtered boxes.
    """
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes >= limit_range[0:3]) &
            (new_boxes <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return boxes[mask], mask


def mask_boxes_outside_range_torch(boxes, lidar_range):
    in_range = (boxes[:, 0] > lidar_range[0]) & \
               (boxes[:, 0] < lidar_range[3]) & \
               (boxes[:, 1] > lidar_range[1]) & \
               (boxes[:, 1] < lidar_range[4])
    return in_range


def remove_points_in_boxes3d(points, boxes3d, x_idx=0):
    """
    :param points: (num_points, x_idx + 3 + C)
    :param boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    :return:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    points, is_numpy = check_numpy_to_torch(points)
    point_masks = points_in_boxes_cpu(points[:, x_idx:x_idx+3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    :param boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def convert_box_to_polygon(boxes_array):
    """
    Convert boxes array to shapely.geometry.Polygon format.

    :param boxes_array : np.ndarray
        (N, 4, 2) or (N, 8, 3).

    :return:
        list of converted shapely.geometry.Polygon object.

    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def compute_iou(box, boxes):
    """
    Compute iou between box and boxes list

    :param box: shapely.geometry.Polygon
        Bounding box Polygon.

    :param boxes: list
        List of shapely.geometry.Polygon.

    :return: iou : np.ndarray
        Array of iou between box and boxes.

    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def bbox_cxcywh_to_xyxy(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    :param bbox (Tensor): Shape (n, 4) for bboxes.
    :return: Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    :param bbox (Tensor): Shape (n, 4) for bboxes.

    :return: Tensor, Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return torch.cat(bbox_new, dim=-1)


def transform_boxes_3d(boxes_in, transform, mode=7):
    """
    :param boxes_in: (N, 7)
    :param transform: (4, 4)
    :param mode: 7 | 9
    """
    is_numpy = isinstance(boxes_in, np.ndarray)
    assert mode == 11 or mode == 9 or mode == 7
    assert boxes_in.shape[-1] == 11 or boxes_in.shape[-1] == 9 or boxes_in.shape[-1] == 7
    if boxes_in.shape[-1] == 11:
        boxes = boxes_in[:, [2, 3, 4, 5, 6, 7, 10]]
    elif boxes_in.shape[-1] == 9:
        boxes = boxes_in[:, [0, 1, 2, 3, 4, 5, 8]]
    else:
        boxes = boxes_in
    boxes_corner = boxes_to_corners_3d(boxes[:, :7])  # (N, 8, 3)
    boxes_corner = boxes_corner.reshape(-1, 3).T  # (N*8, 3)
    if is_numpy:
        boxes_corner = np.concatenate([boxes_corner, np.ones_like(boxes_corner[:1])], axis=0)
    else:
        boxes_corner = torch.cat([boxes_corner, torch.ones_like(boxes_corner[:1])], dim=0)
    # rotate bbx to augmented coords
    boxes_corner = (transform @ boxes_corner)[:3].T.reshape(len(boxes), 8, 3)
    if mode == 11:
        boxes_ = corners_to_boxes_3d(boxes_corner, mode=9)
        if is_numpy:
            boxes = np.concatenate([boxes_in[:, :2], boxes_], axis=-1)
        else:
            boxes = torch.cat([boxes_in[:, :2], boxes_], dim=-1)
    else:
        boxes = corners_to_boxes_3d(boxes_corner, mode=mode)
    if is_numpy and isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    return boxes


def normalize_bbox(bboxes):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, cz, w, l, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    w = normalized_bboxes[..., 3:4]
    l = normalized_bboxes[..., 4:5]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
         # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


if __name__=="__main__":
    boxes = np.random.random((1, 9))
    boxes[:, 3] *= 4
    boxes[:, 4] *= 1.8
    boxes[:, 5] *= 1.6
    boxes[:, 8] *= 3.14

    boxes_corner = boxes_to_corners_3d(boxes)
    boxes_center = corners_to_boxes_3d(boxes_corner)
    print(boxes)
    print(boxes_center)
    print('------------------------------')
    print(boxes_center - boxes)
