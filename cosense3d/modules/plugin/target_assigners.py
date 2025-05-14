import math
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn
import torch_scatter
from scipy.optimize import linear_sum_assignment

from cosense3d.utils.box_utils import (bbox_xyxy_to_cxcywh,
                                       bbox_cxcywh_to_xyxy,
                                       normalize_bbox,
                                       boxes3d_to_standup_bboxes,
                                       rotate_points_batch)
from cosense3d.utils.pclib import rotate_points_along_z_torch
from cosense3d.utils.iou2d_calculator import bbox_overlaps
from cosense3d.modules.utils.gaussian_utils import gaussian_2d
from cosense3d.modules.utils.gevbev_utils import draw_sample_evis, weighted_mahalanobis_dists
from cosense3d.modules.utils.me_utils import metric2indices, update_me_essentials
from cosense3d.modules.utils.box_coder import build_box_coder
from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from cosense3d.dataset.const import CoSenseBenchmarks as csb
from cosense3d.modules.utils.common import pad_r, pad_l, meshgrid
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses import pred_to_conf_unc
from cosense3d.utils.misc import PI


def sample_mining(scores: torch.Tensor,
                  labels: torch.Tensor,
                  dists=None,
                  sample_mining_thr=0.5,
                  max_sample_ratio=5,
                  max_num_sample=None):
    """
    When only limited numbers of negative targets are sampled for training,
    and the majority of the negative samples are ignored, then there is a
    high probability that hard negative targets are also ignored. This will
    weaken the model to learn from these hard negative targets and generate
    a lot of false positives.
    Therefore, this function mines the samples that have high predictive
    scores as training targets. This function should be used after 'pos_neg_sampling'.

    :param scores: (N1, ...Nk) classification scores/confidences that the
        sample belong to foreground.
    :param labels: (N1..., Nk) class labels, -1 indicates ignore, 0 indicates negative,
        positive numbers indicates classes.
    :param dists: distances.
    :param sample_mining_thr: score threshold for sampling
    :param max_sample_ratio: `n_sample` / `n_pos_sample`
    :param max_num_sample: maximum number of samples.
    :return:
    """
    assert scores.ndim == labels.ndim
    assert scores.shape == labels.shape
    pred_pos = scores > sample_mining_thr
    if dists is not None:
        # only mine points that are not too close to the real positive samples
        pred_pos[dists < 3] = False
    not_cared = labels == -1
    sample_inds = torch.where(torch.logical_and(pred_pos, not_cared))[0]
    n_pos = (labels > 0).sum()
    max_num_sample = int(n_pos * max_sample_ratio) if max_num_sample is None else max_num_sample
    if len(sample_inds) > max_num_sample:
        sample_inds = sample_inds[torch.randperm(len(sample_inds))[:max_num_sample]]
    labels[sample_inds] = 0
    return labels


def pos_neg_sampling(labels: torch.Tensor, pos_neg_ratio: float) -> torch.Tensor:
    """
    Downsample negative targets.

    :param labels: class labels.
    :param pos_neg_ratio: ratio = num_neg_samples / num_pos_samples.
    :return: class labels with -1 labels to be ignored during training.
    """
    pos = labels > 0
    neg = labels == 0
    n_neg_sample = pos.sum(dim=0) * pos_neg_ratio
    if neg.sum() > n_neg_sample:
        neg_inds = torch.where(neg)[0]
        perm = torch.randperm(len(neg_inds))[n_neg_sample:]
        labels[neg_inds[perm]] = -1
    return labels


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner."""

    @abstractmethod
    def assign(self, *args, **kwargs):
        """Assign preds to targets."""


class MatchCost:
    """This class is modified from mmdet."""
    @staticmethod
    def classification(cls_pred: torch.Tensor,
                       gt_labels: torch.Tensor,
                       weight: float=1.0) -> torch.Tensor:
        """

        :param cls_pred: Predicted classification logits, shape
                (num_query, num_class).
        :param gt_labels: Label of `gt_bboxes`, shape (num_gt,).
        :param weight: loss_weight.
        :return: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * weight

    @staticmethod
    def bboxl1(bbox_pred: torch.Tensor,
               gt_bboxes: torch.Tensor,
               weight: float=1.,
               box_format: str='xyxy') -> torch.Tensor:
        """

        :param bbox_pred: Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
        :param gt_bboxes: Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        :param weight: loss_weight.
        :param box_format: 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
        :return: bbox_cost value with weight
        """
        if box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        else:
            raise NotImplementedError
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * weight

    @staticmethod
    def giou(bboxes: torch.Tensor,
             gt_bboxes: torch.Tensor,
             weight: float=1.0):
        """

        :param bboxes: Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
        :param gt_bboxes: Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
        :param weight: loss weight.
        :return: giou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode="giou", is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * weight

    @staticmethod
    def iou(bboxes, gt_bboxes, weight=1.0):
        """See giou"""
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode="iou", is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * weight

    @staticmethod
    def l1(pred, gt, weight=1.0):
        """L1 distance between pred and gt Tensors"""
        cost = torch.cdist(pred, gt, p=1)
        return cost * weight

    @staticmethod
    def binary_focal_loss(cls_pred, gt_labels, weight=1., alpha=0.25, gamma=2, eps=1e-12,):
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (
            1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (
            1 - cls_pred).pow(gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * weight

    @staticmethod
    def focal_loss(cls_pred, gt_labels, weight=1., alpha=0.25, gamma=2, eps=1e-12,):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (
            1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (
            1 - cls_pred).pow(gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * weight

    def build(self, type, **kwargs):
        return partial(getattr(self, type), **kwargs)


class HungarianAssigner2D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost, regression iou cost and center2d l1 cost.
    The assignment is done in the following steps, the order matters.

    1. assign every prediction to -1
    2. compute the weighted costs
    3. do Hungarian matching on CPU based on the costs
    4. assign all to 0 (background) first, then for each matched pair
       between predictions and gts, treat this prediction as foreground
       and assign the corresponding gt index (plus 1) to it.
    """

    def __init__(self,
                 cls_cost=dict(type='classification', weight=1.),
                 reg_cost=dict(type='bboxl1', weight=1.0),
                 iou_cost=dict(type='giou', weight=1.0),
                 centers2d_cost=dict(type='l1', weight=1.0)):
        cost_builder = MatchCost()
        self.cls_cost = cost_builder.build(**cls_cost)
        self.reg_cost = cost_builder.build(**reg_cost)
        self.iou_cost = cost_builder.build(**iou_cost)
        self.centers2d_cost = cost_builder.build(**centers2d_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               pred_centers2d,
               gt_bboxes,
               gt_labels,
               centers2d,
               img_size,
               eps: float = 1e-7
               ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        :param bbox_pred: Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
        :param cls_pred: Predicted classification logits, shape
                [num_query, num_class].
        :param pred_centers2d: prediction 2d center points.
        :param gt_bboxes: ground truth bboxes.
        :param gt_labels: Label of `gt_bboxes`, shape (num_gt,).
            img_size: input image size.
        :param centers2d: 2d center points.
        :param img_size: input image size.
        :param eps: A value added to the denominator for
                numerical stability. Default 1e-7.
        :return:
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, assigned_labels
        img_h, img_w = img_size
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)

        # center2d L1 cost
        normalize_centers2d = centers2d / factor[:, 0:2]
        centers2d_cost = self.centers2d_cost(pred_centers2d, normalize_centers2d)

        # weighted sum of above four costs
        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return num_gts, assigned_gt_inds, assigned_labels


class HungarianAssigner3D(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='focal_loss', weight=1.0),
                 reg_cost=dict(type='l1', weight=1.0),
                 iou_cost=dict(type='iou', weight=1.0)):
        cost_builder = MatchCost()
        self.cls_cost = cost_builder.build(**cls_cost)
        self.reg_cost = cost_builder.build(**reg_cost)
        self.iou_cost = cost_builder.build(**iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               code_weights=None,
               eps=1e-7):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, assigned_labels
            # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes)
        if code_weights is not None:
            bbox_pred = bbox_pred * code_weights
            normalized_gt_bboxes = normalized_gt_bboxes * code_weights

        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # # 5. align matched pred and gt
        # aligned_tgt_boxes = torch.zeros_like(bbox_pred)
        # assign_mask = assigned_gt_inds > 0
        # aligned_tgt_boxes[assign_mask] = normalized_gt_bboxes[assigned_gt_inds[assign_mask] - 1]

        # from projects.utils.vislib import draw_points_boxes_plt
        # vis_boxes_pred = denormalize_bbox(bbox_pred[assign_mask], self.pc_range)[:, :-2]
        # vis_boxes_pred[:, :2] /= code_weights[:2]
        # vis_boxes_gt = denormalize_bbox(aligned_tgt_boxes[assign_mask], self.pc_range)[:, :-2]
        # vis_boxes_gt[:, :2] /= code_weights[:2]
        # draw_points_boxes_plt(
        #     pc_range=51.2,
        #     boxes_pred=vis_boxes_pred.detach().cpu().numpy(),
        #     bbox_pred_label=[str(i) for i in range(vis_boxes_pred.shape[0])],
        #     boxes_gt=vis_boxes_gt.detach().cpu().numpy(),
        #     bbox_gt_label=[str(i) for i in range(vis_boxes_gt.shape[0])],
        #     filename='/home/yuan/Downloads/tmp.png'
        # )

        return num_gts, assigned_gt_inds, assigned_labels


class HeatmapAssigner(BaseAssigner):

    @staticmethod
    def draw_heatmap_gaussian(heatmap, center, radius, k=1):
        """Get gaussian masked heatmap.

        Args:
            heatmap (torch.Tensor): Heatmap to be masked.
            center (torch.Tensor): Center coord of the heatmap.
            radius (int): Radius of gaussian.
            k (int, optional): Multiple of masked_gaussian. Defaults to 1.

        Returns:
            torch.Tensor: Masked heatmap.
        """
        diameter = 2 * radius + 1
        gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom,
                     radius - left:radius + right]).to(heatmap.device,
                                                       torch.float32)
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def assign(self, obj_centers2d, obj_bboxes, img_shape, stride):
        img_h, img_w = img_shape[:2]
        heatmap = torch.zeros(img_h // stride, img_w // stride, device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / 16)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = self.draw_heatmap_gaussian(heatmap, center / 16, radius=int(r), k=1)
        return heatmap


class BoxAnchorAssigner(BaseAssigner, torch.nn.Module):
    def __init__(self,
                 box_size,
                 dirs,
                 voxel_size,
                 lidar_range,
                 stride,
                 box_coder,
                 pos_threshold=0.6,
                 neg_threshold=0.45,
                 score_thrshold=0.25,
                 ):
        super().__init__()
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range
        self.num_anchors = len(dirs)
        self.stride = stride
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.score_thrshold = score_thrshold
        self.box_coder = build_box_coder(**box_coder)
        anchors, standup_anchors = self.get_anchor_template(box_size, dirs)
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.standup_anchors = nn.Parameter(standup_anchors, requires_grad=False)

    def get_anchor_template(self, box_size, dirs):
        pix_x = self.voxel_size[0] * self.stride
        pix_y = self.voxel_size[1] * self.stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y) + pix_y * 0.5
        xys = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        xys = xys.unsqueeze(2).repeat(1, 1, self.num_anchors, 1)
        zs = - torch.ones_like(xys[..., :1])
        h, w = xys.shape[:2]
        lwh = torch.tensor(box_size).reshape(
            1, 1, 1, -1).repeat(h, w, self.num_anchors, 1)
        rs = torch.deg2rad(torch.tensor(dirs)).reshape(
            1, 1, -1, 1).repeat(h, w, 1, 1)
        # (w, h, num_anchor, 7) --> (whn, 7)
        anchors = torch.cat([xys, zs, lwh, rs], dim=-1)
        self.anchor_shape = anchors.shape
        anchors = anchors.view(-1, 7)
        standup_anchors = boxes3d_to_standup_bboxes(anchors)
        return anchors, standup_anchors

    def assign(self, gt_boxes):
        """

        Parameters
        ----------
        gt_boxes Tensor(N, 7): [x, y, z, l, w, h, r, ...]

        Returns
        -------
        reg Tensor(H, W, num_anchors, code_size): box regression targets
        """
        if len(gt_boxes) == 0:
            labels = gt_boxes.new_full((self.standup_anchors.shape[0],), -1)
            reg_tgt = gt_boxes.new_zeros((0, self.box_coder.code_size))
            dir_scores = gt_boxes.new_zeros((0, 4))
            # Todo dir_score, gt_boxes, correct shape
            return labels, reg_tgt, dir_scores

        standup_boxes = boxes3d_to_standup_bboxes(gt_boxes[:, :7])
        ious = self.box_overlaps(self.standup_anchors, standup_boxes)
        iou_max, max_inds = ious.max(dim=1)
        top1_inds = torch.argmax(ious, dim=0)

        pos = iou_max > self.pos_threshold
        pos_inds = torch.cat([top1_inds, torch.where(pos)[0]]).unique()
        neg = iou_max < self.neg_threshold
        neg[pos_inds] = False

        labels = gt_boxes.new_full((ious.shape[0],), -1)
        labels[neg] = 0
        labels[pos_inds] = 1

        aligned_gt_boxes = gt_boxes[max_inds[pos_inds], :7]
        aligned_anchors = self.anchors[pos_inds]
        reg_tgt, dir_score = self.box_coder.encode(aligned_anchors, aligned_gt_boxes)

        return labels, reg_tgt, dir_score

    def box_overlaps(self, boxes1, boxes2):
        areas1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * \
                 (boxes1[:, 3] - boxes1[:, 1] + 1)
        areas2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * \
                 (boxes2[:, 3] - boxes2[:, 1] + 1)

        boxes1_mat = boxes1.unsqueeze(1).repeat(1, boxes2.shape[0], 1)
        boxes2_mat = boxes2.unsqueeze(0).repeat(boxes1.shape[0], 1, 1)
        x_extend = torch.minimum(boxes1_mat[..., 2], boxes2_mat[..., 2]) - \
            torch.maximum(boxes1_mat[..., 0], boxes2_mat[..., 0]) + 1
        y_extend = torch.minimum(boxes1_mat[..., 3], boxes2_mat[..., 3]) - \
            torch.maximum(boxes1_mat[..., 1], boxes2_mat[..., 1]) + 1

        overlaps = torch.zeros_like(boxes1_mat[..., 0])

        pos = torch.logical_and(x_extend > 0, y_extend > 0)
        intersection = x_extend[pos] * y_extend[pos]
        union = (areas1.unsqueeze(1) + areas2.unsqueeze(0))[pos] - intersection
        overlaps[pos] = intersection / union

        return overlaps

    def get_predictions(self, preds):
        # roi = {'box': [], 'scr': [], 'lbl': [], 'idx': []}
        roi = {}
        B = len(preds['cls'])
        pred_cls = preds['cls'].sigmoid().permute(0, 3, 2, 1).reshape(B, -1)
        pred_reg = preds['reg'].permute(0, 3, 2, 1).reshape(B, -1, 7)
        indices = torch.stack([torch.ones_like(pred_cls[0]) * i for i in range(B)], dim=0)

        anchors = self.anchors.unsqueeze(0).repeat(B, 1, 1)
        pos = pred_cls > self.score_thrshold

        boxes_dec = self.box_coder.decode(anchors, pred_reg)
        # remove abnormal boxes
        mask = (boxes_dec[..., 3:6] > 0.1) & (boxes_dec[..., 3:6] < 10)
        pos = torch.logical_and(pos, mask.all(dim=-1))

        pred_cls = pred_cls[pos]
        pred_box = boxes_dec[pos]
        roi['scr'] = pred_cls
        roi['box'] = pred_box
        # TODO currently only support class car
        roi['lbl'] = torch.zeros_like(pred_cls)
        roi['idx'] = indices[pos]

        return roi


class BoxAnchorAssignerSecond(BaseAssigner, torch.nn.Module):
    def __init__(self,
                 box_size,
                 dirs,
                 voxel_size,
                 lidar_range,
                 stride,
                 box_coder,
                 pos_threshold=0.6,
                 neg_threshold=0.45,
                 score_threshold=0.25,
                 ):
        super().__init__()
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range
        self.num_anchors = len(dirs)
        self.box_size = box_size
        self.dirs = dirs
        self.stride = stride
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.score_threshold = score_threshold
        self.box_coder = build_box_coder(**box_coder)

    def get_anchors(self, ctrs):
        xys = ctrs[:, :2].unsqueeze(1).repeat(1, self.num_anchors, 1)
        zs = - torch.ones_like(xys[..., :1])
        n = xys.shape[0]
        lwh = torch.tensor(self.box_size).reshape(
            1, 1, -1).repeat(n, self.num_anchors, 1).to(ctrs.device)
        rs = torch.deg2rad(torch.tensor(self.dirs)).reshape(
            1, -1, 1).repeat(n, 1, 1).to(ctrs.device)
        # (n, num_anchor, 7) --> (whn, 7)
        anchors = torch.cat([xys, zs, lwh, rs], dim=-1)
        anchors = anchors.view(-1, 7)
        return anchors

    def assign(self, ctrs, gt_boxes, **kwargs):
        """

        Parameters
        ----------
        ctrs Tensor(N, 2): [x, y]
        gt_boxes Tensor(M, 7+C): [x, y, z, l, w, h, r, ...]

        Returns
        -------
        reg Tensor(H, W, num_anchors, code_size): box regression targets
        """
        if len(gt_boxes) == 0:
            labels = gt_boxes.new_full((ctrs.shape[0]*2,), -1)
            reg_tgt = gt_boxes.new_zeros((0, self.box_coder.code_size+2))
            dir_scores = gt_boxes.new_zeros((0, 4))
            # Todo dir_score, gt_boxes, correct shape
            return labels, reg_tgt, dir_scores

        anchors = self.get_anchors(ctrs)
        ious = boxes_iou_bev(anchors, gt_boxes[:, :7])
        iou_max, max_inds = ious.max(dim=1)

        pos = iou_max > self.pos_threshold
        pos_inds = torch.where(pos)[0]
        neg = iou_max < self.neg_threshold
        neg[pos_inds] = False

        labels = gt_boxes.new_full((ious.shape[0],), -1)
        labels[neg] = 0
        labels[pos_inds] = 1

        aligned_gt_boxes = gt_boxes[max_inds[pos_inds]]
        aligned_anchors = anchors[pos_inds]
        reg_tgt, dir_score = self.box_coder.encode(aligned_anchors, aligned_gt_boxes[:, :7])
        reg_tgt = torch.cat([reg_tgt, aligned_gt_boxes[:, 7:]], dim=-1).contiguous()

        return labels, reg_tgt, dir_score

    def get_predictions(self, ctrs, preds):
        # roi = {'box': [], 'scr': [], 'lbl': [], 'idx': []}
        roi = {}
        B = len(preds['cls'])
        pred_cls = preds['cls'].sigmoid().reshape(-1)
        pred_reg = preds['reg'].reshape(-1, 7)

        anchors = self.get_anchors(preds['ctr'])
        pos = pred_cls > self.score_threshold

        boxes_dec = self.box_coder.decode(anchors, pred_reg)
        # remove abnormal boxes
        mask = (boxes_dec[..., 3:6] > 0.1) & (boxes_dec[..., 3:6] < 10)
        pos = torch.logical_and(pos, mask.all(dim=-1))

        pred_cls = pred_cls[pos]
        pred_box = boxes_dec[pos]
        inds = ctrs[:, :1].repeat(1, 2).view(-1)[pos]
        roi['scr'] = pred_cls
        roi['box'] = pred_box
        # TODO currently only support class car
        roi['lbl'] = torch.zeros_like(pred_cls)
        roi['idx'] = inds

        return roi


class BoxSparseAnchorAssigner(BaseAssigner, torch.nn.Module):
    def __init__(self,
                 box_size,
                 dirs,
                 voxel_size,
                 lidar_range,
                 stride,
                 box_coder,
                 me_coor=True,
                 pos_threshold=0.6,
                 neg_threshold=0.45,
                 score_thrshold=0.25,
                 ):
        super().__init__()
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range
        self.num_anchors = len(dirs)
        self.stride = stride
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.score_thrshold = score_thrshold
        self.box_coder = build_box_coder(**box_coder)
        anchors, standup_anchors = self.get_anchor_template(box_size, dirs)
        self.anchors = nn.Parameter(anchors, requires_grad=False)
        self.standup_anchors = nn.Parameter(standup_anchors, requires_grad=False)
        if me_coor:
            lr = lidar_range
            res_x, res_y = stride * voxel_size[0], stride * voxel_size[1]
            self.size_x = round((lr[3] - lr[0]) / res_x)
            self.size_y = round((lr[4] - lr[1]) / res_y)
            self.offset_sz_x = round(lr[0] / res_x)
            self.offset_sz_y = round(lr[1] / res_y)
            self.coor_to_inds = self.me_coor_to_grid_indices
        else:
            raise NotImplementedError

    def me_coor_to_grid_indices(self, coor):
        inds = coor / self.stride
        inds[:, 0] -= self.offset_sz_x
        inds[:, 1] -= self.offset_sz_y
        in_range_mask = (inds >= 0).all(dim=-1) & (inds[:, 0] < self.size_x) & (inds[:, 1] < self.size_y)
        return inds[in_range_mask].long(), in_range_mask

    def get_anchor_template(self, box_size, dirs):
        pix_x = self.voxel_size[0] * self.stride
        pix_y = self.voxel_size[1] * self.stride
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], pix_x) + pix_x * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], pix_y) + pix_y * 0.5
        xys = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1)
        xys = xys.unsqueeze(2).repeat(1, 1, self.num_anchors, 1)
        zs = - torch.ones_like(xys[..., :1])
        h, w = xys.shape[:2]
        lwh = torch.tensor(box_size).reshape(
            1, 1, 1, -1).repeat(h, w, self.num_anchors, 1)
        rs = torch.deg2rad(torch.tensor(dirs)).reshape(
            1, 1, -1, 1).repeat(h, w, 1, 1)
        # (w, h, num_anchor, 7) --> (whn, 7)
        anchors = torch.cat([xys, zs, lwh, rs], dim=-1)
        standup_anchors = boxes3d_to_standup_bboxes(
            anchors.view(-1, 7)).reshape(h, w, self.num_anchors, 4)
        return anchors, standup_anchors

    def assign(self, coors: torch.Tensor, gt_boxes: torch.Tensor):
        """

        :param coors: (N, 2) 2D mink coor [x, y]
        :param gt_boxes: (M, 7) [x, y, z, l, w, h, r]
        :return:
            - labels Tensor(N, num_anchors): box regression targets
            - reg_tgt Tensor(N, num_anchors, code_size): box regression targets
            - ir_score Tensor(N, num_anchors, 4) or None: direction score target
        """
        gt_boxes = gt_boxes[:, :7]
        if len(gt_boxes) == 0:
            labels = gt_boxes.new_full((coors.shape[0] * self.num_anchors,), -1)
            reg_tgt = gt_boxes.new_zeros((0, self.box_coder.code_size))
            dir_scores = gt_boxes.new_zeros((0, 4))
            # Todo dir_score, gt_boxes, correct shape
            return labels, reg_tgt, dir_scores
        inds, in_range_mask = self.coor_to_inds(coors)
        gt_standup_boxes = boxes3d_to_standup_bboxes(gt_boxes)
        standup_anchors = self.standup_anchors[inds[:, 0], inds[:, 1]].view(-1, 4)
        ious = self.box_overlaps(standup_anchors, gt_standup_boxes)
        iou_max, max_inds = ious.max(dim=1)
        top1_inds = torch.argmax(ious, dim=0)

        pos = iou_max > self.pos_threshold
        pos_inds = torch.cat([top1_inds, torch.where(pos)[0]]).unique()
        neg = iou_max < self.neg_threshold
        neg[pos_inds] = False

        labels = gt_boxes.new_full((ious.shape[0],), -1)
        labels[neg] = 0
        labels[pos_inds] = 1

        aligned_gt_boxes = gt_boxes[max_inds[pos_inds]]
        aligned_anchors = self.anchors[inds[:, 0], inds[:, 1]].view(-1, self.box_coder.code_size)[pos_inds]
        reg_tgt, dir_score = self.box_coder.encode(aligned_anchors, aligned_gt_boxes)

        labels_final = gt_boxes.new_full((in_range_mask.shape[0], self.num_anchors), -1)
        labels_final[in_range_mask] = labels.view(-1, self.num_anchors)
        return labels_final.view(-1), reg_tgt, dir_score

    def box_overlaps(self, boxes1, boxes2):
        areas1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * \
                 (boxes1[:, 3] - boxes1[:, 1] + 1)
        areas2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * \
                 (boxes2[:, 3] - boxes2[:, 1] + 1)

        boxes1_mat = boxes1.unsqueeze(1).repeat(1, boxes2.shape[0], 1)
        boxes2_mat = boxes2.unsqueeze(0).repeat(boxes1.shape[0], 1, 1)
        x_extend = torch.minimum(boxes1_mat[..., 2], boxes2_mat[..., 2]) - \
            torch.maximum(boxes1_mat[..., 0], boxes2_mat[..., 0]) + 1
        y_extend = torch.minimum(boxes1_mat[..., 3], boxes2_mat[..., 3]) - \
            torch.maximum(boxes1_mat[..., 1], boxes2_mat[..., 1]) + 1

        overlaps = torch.zeros_like(boxes1_mat[..., 0])

        pos = torch.logical_and(x_extend > 0, y_extend > 0)
        intersection = x_extend[pos] * y_extend[pos]
        union = (areas1.unsqueeze(1) + areas2.unsqueeze(0))[pos] - intersection
        overlaps[pos] = intersection / union

        return overlaps

    def get_predictions(self, coors, preds):
        """

        :param coors: Tensor(N, 3) mink coor [batch_idx, x, y]
        :param preds:
        :return:
        """
        # roi = {'box': [], 'scr': [], 'lbl': [], 'idx': []}
        roi = {}
        inds, in_range_mask = self.coor_to_inds(coors[:, 1:])
        pred_cls = preds['cls'][in_range_mask].sigmoid().reshape(-1)
        pred_reg = preds['reg'][in_range_mask].reshape(-1, 7)
        indices = coors[:, 0:1][in_range_mask].repeat(1, self.num_anchors).reshape(-1)

        anchors = self.anchors[inds[:, 0], inds[:, 1]].view(-1, self.box_coder.code_size)
        pos = pred_cls > self.score_thrshold
        anchors = anchors[pos]
        pred_cls = pred_cls[pos]
        pred_reg = pred_reg[pos]
        indices = indices[pos]

        boxes_dec = self.box_coder.decode(anchors, pred_reg)

        # remove abnormal boxes
        mask = (boxes_dec[..., 3:6] > 0.1) & (boxes_dec[..., 3:6] < 10)
        mask = mask.all(dim=-1)
        pred_cls = pred_cls[mask]
        pred_box = boxes_dec[mask]
        indices = indices[mask]

        roi['scr'] = pred_cls
        roi['box'] = pred_box
        # TODO currently only support class car
        roi['lbl'] = torch.zeros_like(pred_cls)
        roi['idx'] = indices

        return roi


class BoxCenterAssigner(BaseAssigner, torch.nn.Module):
    def __init__(self,
                 voxel_size,
                 lidar_range,
                 stride,
                 detection_benchmark,
                 class_names_each_head,
                 center_threshold,
                 box_coder,
                 activation='relu',
                 edl=True,
                 ):
        super().__init__()
        self.voxel_size = voxel_size
        self.lidar_range = lidar_range
        self.meter_per_pixel = (voxel_size[0] * stride, voxel_size[1] * stride)
        self.csb = csb.get(detection_benchmark)
        self.class_names_each_head = class_names_each_head
        self.activation = activation
        self.center_threshold = center_threshold
        self.box_coder = build_box_coder(**box_coder)
        self.edl = edl

    def pts_to_indices(self, bev_pts: torch.Tensor):
        """
        :param bev_pts: (N, 3+), BEV points, 1st column should be batch index.
        :return:
        """
        x = (bev_pts[:, 1] - self.meter_per_pixel[0] * 0.5 - self.lidar_range[0]) \
                  / self.meter_per_pixel[0]
        y = (bev_pts[:, 2] - self.meter_per_pixel[1] * 0.5 - self.lidar_range[1]) \
                  / self.meter_per_pixel[1]
        indices = torch.stack([bev_pts[:, 0].long(), x.long(), y.long()], dim=1)
        return indices

    @torch.no_grad()
    def assign(self, centers, gt_boxes, gt_labels, gt_preds=None, **kwargs):
        box_names = [self.csb[c.item()][0] for c in gt_labels]

        # cal regression targets
        reg_tgt = {'box': [], 'dir': [], 'scr': [], 'idx': [], 'valid_mask': [], 'vel': [], 'pred': []}
        for h, cur_cls_names in enumerate(self.class_names_each_head):
            center_indices = self.pts_to_indices(centers).T
            box_mask = [n in cur_cls_names for n in box_names]
            cur_boxes = gt_boxes[box_mask]
            res = self.box_coder.encode(centers, cur_boxes, self.meter_per_pixel, gt_preds)
            reg_box, reg_dir, dir_score, valid = res[:4]

            reg_tgt['idx'].append(center_indices[:, valid])
            reg_tgt['valid_mask'].append(valid)
            reg_tgt['box'].append(reg_box)
            reg_tgt['dir'].append(reg_dir)
            reg_tgt['scr'].append(dir_score)
            if getattr(self.box_coder, 'with_velo', False):
                reg_tgt['vel'].append(res[4])
            if getattr(self.box_coder, 'with_pred', False):
                reg_tgt['pred'].append(res[5])
        return reg_tgt

    def get_predictions(self, preds):
        """Decode the center and regression maps into BBoxes.

        :param preds:
            - cls: list[Tensor], each tensor is the result from a cls head with shape (B or N, Ncls, ...).
            - reg:
                * box: list[Tensor], one tensor per reg head with shape (B or N, 6, ...).
                * dir: list[Tensor], one tensor per reg head with shape (B or N, 8, ...).
                * scr: list[Tensor], one tensor per reg head with shape (B or N, 4, ...).
        :return: roi:
                * box: list[Tensor], one tensor per head with shape (N, 8).
                * scr: list[Tensor], one tensor per head with shape (N,).
                * lbl: list[Tensor], one tensor per head with shape (N,).
                * idx: list[Tensor], one tensor per head with shape (3, N), center map indices of the boxes.
        """
        roi = {'box': [], 'scr': [], 'lbl': [], 'idx': []}
        lbl_cnt = torch.cumsum(torch.Tensor([0] + [m.shape[1] for m in preds['cls']]), dim=0)
        confs = []
        for h, center_cls in enumerate(preds['cls']):
            if center_cls.ndim == 4:
                conf, _ = pred_to_conf_unc(center_cls.permute(0, 2, 3, 1), self.activation)
                center_mask = conf[..., 1:].max(dim=-1).values > self.center_threshold  # b, h, w
                center_indices = torch.stack(torch.where(center_mask), dim=0)
                centers = self.indices_to_pts(center_indices[1:]).T
                cur_centers = torch.cat([center_indices[0].unsqueeze(-1), centers], dim=-1)
                cur_reg = {k: preds['reg'][k][h].permute(0, 2, 3, 1)[center_mask]
                           for k in ['box', 'dir', 'scr']}
            else:
                conf, _ = pred_to_conf_unc(center_cls, self.activation, self.edl)
                centers = preds['ctr']
                if self.edl:
                    center_mask = conf[..., 1:].max(dim=-1).values > self.center_threshold  # b, h, w
                else:
                    center_mask = conf.max(dim=-1).values > self.center_threshold  # b, h, w

                if center_cls.ndim == 3:
                    indices = torch.stack([torch.zeros_like(centers[i, :, :1]) + i for i in range(centers.shape[0])], dim=0)
                    centers = torch.cat([indices, centers], dim=-1)

                cur_centers = centers[center_mask]
                center_indices = self.pts_to_indices(cur_centers)
                cur_reg = {k: preds['reg'][k][h][center_mask]
                           for k in preds['reg'].keys()}

                # from cosense3d.utils import vislib
                # mask = cur_centers[:, 0].int() == 0
                # confs = conf[center_mask][mask, 1].detach().cpu().numpy()
                # points = cur_centers[mask, 1:].detach().cpu().numpy()
                # fig = vislib.plt.figure(figsize=(6, 6))
                # vislib.plt.scatter(points[:, 0], points[:, 1], c=confs, s=1)
                # vislib.plt.show()
                # vislib.plt.close()

            cur_box = self.box_coder.decode(cur_centers, cur_reg)
            cur_scr, cur_lbl = conf[center_mask].max(dim=-1)
            cur_lbl = cur_lbl + lbl_cnt[h]
            roi['box'].append(cur_box)
            roi['scr'].append(cur_scr)
            roi['lbl'].append(cur_lbl)
            roi['idx'].append(center_indices)
            confs.append(conf)

            # from cosense3d.utils.vislib import draw_points_boxes_plt
            # points = centers[:, 1:].detach().cpu().numpy()
            # boxes = cur_box[:, 1:].detach().cpu().numpy()
            # draw_points_boxes_plt(
            #     pc_range=self.lidar_range,
            #     boxes_pred=boxes,
            #     points=points,
            #     filename="/home/yuan/Pictures/tmp.png"
            # )

        # merge detections from all heads
        roi['box'] = torch.cat(roi['box'], dim=0)
        roi['scr'] = torch.cat(roi['scr'], dim=0)
        roi['lbl'] = torch.cat(roi['lbl'], dim=0)
        roi['idx'] = torch.cat(roi['idx'], dim=0)
        confs = torch.stack(confs, dim=1)
        return roi, confs


class BEVCenternessAssigner(BaseAssigner):
    """
    Assign center points in the BEV maps to positve if the point is in the range 'min_radius' of any gt box center.
    """
    def __init__(self,
                 n_cls,
                 min_radius=1.0,
                 pos_neg_ratio=5,
                 mining_thr=0,
                 max_mining_ratio=3,
                 mining_start_epoch=5,
                 merge_all_classes=False,
                 use_gaussian=False,
                 sigma=1.0
                 ):
        super().__init__()
        self.n_cls = n_cls
        self.min_radius = min_radius
        self.pos_neg_ratio = pos_neg_ratio
        self.sample_mining_thr = mining_thr
        self.max_mining_ratio = max_mining_ratio
        self.mining_start_epoch = mining_start_epoch
        self.merge_all_classes = merge_all_classes
        self.use_gaussian = use_gaussian
        self.sigma = sigma

    def get_labels_single_head(self, centers, gt_boxes, pred_scores=None, **kwargs):
        diff = centers[:, :2].unsqueeze(1) - gt_boxes[:, :2].unsqueeze(0)
        dists = torch.norm(diff, dim=-1)
        dists_min, dists_min_arg = dists.min(dim=1)
        if self.use_gaussian:
            labels = torch.exp(-0.5 * torch.sqrt(dists_min) / self.sigma ** 2)
            # sigmas = gt_boxes[:, 3:5][dists_min_arg] / 4 * self.sigma
            # labels = weighted_mahalanobis_dists(
            #     sigmas ** 2, diff[torch.arange(len(diff)), dists_min_arg].abs().unsqueeze(1))
            labels[labels < 1e-4] = 0
        else:
            labels = (dists_min < self.min_radius).float()

        if self.pos_neg_ratio:
            labels = pos_neg_sampling(labels, self.pos_neg_ratio)
        if self.sample_mining_thr > 0 and kwargs.get('epoch', 0) > self.mining_start_epoch:
            assert pred_scores is not None
            labels = sample_mining(pred_scores, labels,
                                   dists_min,
                                   self.sample_mining_thr,
                                   self.max_mining_ratio)

        return labels

    @torch.no_grad()
    def assign(self, centers, gt_boxes, gt_labels, pred_scores=None, **kwargs):
        if len(gt_boxes) == 0:
            labels = torch.zeros_like(centers[:, :1])
            return labels
        if self.merge_all_classes:
            labels = self.get_labels_single_head(centers, gt_boxes).unsqueeze(-1)
        else:
            labels = []
            for n in range(self.n_cls):
                cur_boxes = gt_boxes[gt_labels == n]
                cur_scores = None if pred_scores is None else pred_scores[n]
                labels.append(self.get_labels_single_head(centers, cur_boxes, cur_scores, **kwargs))
            labels = torch.stack(labels, dim=-1)

        # import matplotlib.pyplot as plt
        #
        # from cosense3d.utils import vislib
        # pc_range = [-100, -41.6, -3.0, 100, 41.6, 3.0]
        # label = labels.detach().cpu().numpy()
        # label = label[:, 0]
        # points = centers.detach().cpu().numpy()
        # boxes = gt_boxes.cpu().numpy()
        # ax = vislib.draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     boxes_gt=boxes,
        #     return_ax=True
        # )
        # ax.scatter(points[:, 0], points[:, 1], cmap='jet', c=label, s=1)
        # plt.savefig("/home/yuan/Downloads/tmp.png")
        # plt.close()

        return labels


class BEVBoxAssigner(BaseAssigner):
    """
    Assign center points in the BEV maps to positve if the point is in the range 'min_radius' of any gt box center.
    """
    def __init__(self,
                 n_cls,
                 pos_neg_ratio=5,
                 mining_thr=0,
                 max_mining_ratio=3,
                 mining_start_epoch=5,
                 merge_all_classes=False,
                 ):
        super().__init__()
        self.n_cls = n_cls
        self.pos_neg_ratio = pos_neg_ratio
        self.sample_mining_thr = mining_thr
        self.max_mining_ratio = max_mining_ratio
        self.mining_start_epoch = mining_start_epoch
        self.merge_all_classes = merge_all_classes

    def get_labels_single_head(self, centers, gt_boxes, pred_scores=None, **kwargs):
        boxes = pad_l(gt_boxes[:, :7]).clone()
        boxes[:, 3] = 0
        pts = pad_r(pad_l(centers[:, :2]))

        _, box_idx_of_pts = points_in_boxes_gpu(
            pts, boxes, batch_size=1
        )
        labels = (box_idx_of_pts >= 0).float()
        if self.pos_neg_ratio:
            labels = pos_neg_sampling(labels, self.pos_neg_ratio)

        return labels

    @torch.no_grad()
    def assign(self, centers, gt_boxes, gt_labels, pred_scores=None, **kwargs):
        if len(gt_boxes) == 0:
            labels = torch.zeros_like(centers[:, :1])
            return labels
        if self.merge_all_classes:
            labels = self.get_labels_single_head(centers, gt_boxes).unsqueeze(-1)
        else:
            labels = []
            for n in range(self.n_cls):
                cur_boxes = gt_boxes[gt_labels == n]
                cur_scores = None if pred_scores is None else pred_scores[n]
                labels.append(self.get_labels_single_head(centers, cur_boxes, cur_scores, **kwargs))
            labels = torch.stack(labels, dim=-1)

        # import matplotlib.pyplot as plt
        #
        # from cosense3d.utils import vislib
        # pc_range = [-100, -41.6, -3.0, 100, 41.6, 3.0]
        # label = labels.detach().cpu().numpy()
        # label = label[:, 0]
        # points = centers.detach().cpu().numpy()
        # boxes = gt_boxes.cpu().numpy()
        # ax = vislib.draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     boxes_gt=boxes,
        #     return_ax=True
        # )
        # ax.scatter(points[:, 0], points[:, 1], cmap='jet', c=label, s=1)
        # plt.savefig("/home/yuan/Downloads/tmp.png")
        # plt.close()

        return labels


class BEVPointAssigner(BaseAssigner):
    """
    Assign target points to BEV boxes and down-sample the target points with buffered-based method.
    """
    def __init__(self,
                 down_sample=True,
                 sample_mining_thr=0.,
                 max_mining_ratio=3,
                 annealing_step=None,
                 topk_sampling=False,
                 annealing_sampling=False,
                 ):
        super().__init__()
        self.down_sample = down_sample
        self.sample_mining_thr = sample_mining_thr
        self.max_mining_ratio = max_mining_ratio
        self.annealing_step = annealing_step
        self.topk_sampling = topk_sampling
        self.annealing_sampling = annealing_sampling

    def downsample_tgt_pts(self, tgt_label, max_sam):
        selected = torch.ones_like(tgt_label.bool())
        pos = tgt_label == 1
        if pos.sum() > max_sam:
            mask = torch.rand_like(tgt_label[pos].float()) < max_sam / pos.sum()
            selected[pos] = mask

        buffer = tgt_label == 0
        if buffer.sum() > max_sam:
            mask = torch.rand_like(tgt_label[buffer].float()) < max_sam / buffer.sum()
            selected[buffer] = mask

        neg = tgt_label == -1
        if neg.sum() > max_sam:
            mask = torch.rand_like(tgt_label[neg].float()) < max_sam / neg.sum()
            selected[neg] = mask
            labels = - torch.ones_like(mask).long()
            labels[mask] = 0
            tgt_label[neg] = labels
        return selected, tgt_label

    def assign(self, tgt_pts, gt_boxes, B, conf=None, down_sample=True, **kwargs):
        boxes = gt_boxes.clone()
        boxes[:, 3] = 0
        pts = pad_r(tgt_pts)

        if not down_sample or not self.down_sample:
            _, box_idx_of_pts = points_in_boxes_gpu(
                pts, boxes, batch_size=B
            )
            tgt_label = torch.zeros_like(box_idx_of_pts)
            tgt_label[box_idx_of_pts >= 0] = 1
            return tgt_pts, tgt_label, None

        _, box_idx_of_pts = points_in_boxes_gpu(
            pts, boxes, batch_size=B
        )
        boxes[:, 4:6] *= 2
        _, enlarged_box_idx_of_pts = points_in_boxes_gpu(
            pts, boxes, batch_size=B
        )

        pos_mask = box_idx_of_pts >= 0
        buffer_mask = (box_idx_of_pts < 0) & (enlarged_box_idx_of_pts >= 0)
        tgt_label = - torch.ones_like(box_idx_of_pts)
        tgt_label[pos_mask] = 1
        tgt_label[buffer_mask] = 0
        n_sam = len(boxes) * 50

        # add points that have high pred scores
        if self.sample_mining_thr > 0:
            scores = conf[..., 1:].sum(dim=-1)
            tgt_label = sample_mining(scores, tgt_label, self.sample_mining_thr,
                                      max_num_sample=n_sam)

        mask, tgt_label = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)

        # get final tgt
        tgt_pts = tgt_pts[mask]
        tgt_label = tgt_label[mask]

        return tgt_pts, tgt_label, mask

    def get_predictions(self, x, edl=True, activation='none'):
        conf, unc = pred_to_conf_unc(x, activation, edl)
        return conf, unc


class BEVSemsegAssigner(BaseAssigner):
    def __init__(self,
                 data_info,
                 stride,
                 tgt_range=None,
                 down_sample=False,
                 annealing_step=None,
                 ):
        super().__init__()
        update_me_essentials(self, data_info, stride)
        self.tgt_range = tgt_range
        self.downsample = down_sample
        self.annealing_step = annealing_step

    def pts_to_inds(self, pts):
        """Calculate indices of samples in the bev map"""
        ixy = metric2indices(pts[:, :3], self.res).long()
        ixy[:, 1] -= self.offset_sz_x
        ixy[:, 2] -= self.offset_sz_y
        maskx = torch.logical_and(ixy[:, 1] >= 0, ixy[:, 1] < self.size_x)
        masky = torch.logical_and(ixy[:, 2] >= 0, ixy[:, 2] < self.size_y)
        mask = torch.logical_and(maskx, masky)
        indices = ixy[mask]
        return indices.T, mask

    def get_obs_mask(self, inds, B):
        obs_mask = torch.zeros((B, self.size_x, self.size_y), device=inds.device)
        inds = inds.clone().long().T
        inds[1] -= self.offset_sz_x
        inds[2] -= self.offset_sz_y
        obs_mask[inds[0], inds[1], inds[2]] = 1
        return obs_mask.bool()

    @staticmethod
    def down_sample_pred_pts(ctr_pts):
        keep = torch.rand_like(ctr_pts['ctr'][:, 0]) > 0.5
        for k in ctr_pts.keys():
            ctr_pts[k] = ctr_pts[k][keep]

        return ctr_pts

    @torch.no_grad()
    def downsample_tgt_pts(self, tgt_label, max_sam):
        selected = torch.ones_like(tgt_label.bool())
        pos = tgt_label == 1
        if pos.sum() > max_sam:
            mask = torch.rand_like(tgt_label[pos].float()) < max_sam / pos.sum()
            selected[pos] = mask

        neg = tgt_label == 0
        if neg.sum() > max_sam:
            mask = torch.rand_like(tgt_label[neg].float()) < max_sam / neg.sum()
            selected[neg] = mask
        return selected

    def filter_range(self, ctr_pts, samples):
        mask = (ctr_pts['ctr'].abs() < self.tgt_range).all(1)
        for k in ctr_pts.keys():
            ctr_pts[k] = ctr_pts[k][mask]

        mask = (samples[:, 1:3].abs() < self.tgt_range).all(1)
        samples = samples[mask]
        return ctr_pts, samples

    def assign(self, ctr_pts, samples, B, gt_boxes=None, **kwargs):
        raise NotImplementedError

    def get_predictions(self, data_dict, B, edl=True, activation='none', **kwargs):
        raise NotImplementedError


class ContiBEVAssigner(BEVSemsegAssigner):
    def __init__(self,
                 distr_r=2.0,
                 var0=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.distr_r = distr_r
        self.var0 = var0
        steps = int(self.distr_r / self.res[0]) * 2 + 1
        offset = meshgrid(-self.distr_r, self.distr_r, 2,
                          n_steps=steps).cuda().view(-1, 2)
        self.nbrs = offset[torch.norm(offset, dim=1) < 2].view(1, -1, 2)

    def sample_dynamic_tgt_pts(self, ctr_pts: dict, gt_boxes: torch.Tensor, B: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the input coordinates of the center points and the ground truth BBoxes,
        sample the BEV target points for BEV semantic segmentation following the buffer-based sampling as illustrated
        in the following image:

        .. image:: _static/imgs/buffer_based_sampling.png
          :width: 400
          :alt: Buffer-based sampling of the BEV target

        :param ctr_pts: center points of bev maps, including indices, metric centers and regression results.
        :param gt_boxes: ground truth BBoxes.
        :param B: batch size.
        :return:
             - tgt_pts: sampled target points.
             - tgt_lbl: labels of the sampled target points.
             - inds: map indices of the sampled target points.
        """
        tgt_pts = ctr_pts['ctr'].clone()
        tgt_pts[:, :2] = tgt_pts[:, :2] + torch.randn_like(tgt_pts[:, :2]) * 3
        tgt_pts = torch.cat([ctr_pts['coor'][:, :1], tgt_pts], dim=-1)
        obs_mask = self.get_obs_mask(ctr_pts['coor'], B)
        inds, mask = self.pts_to_inds(tgt_pts)
        tgt_pts = tgt_pts[mask]
        mask = obs_mask[inds[0], inds[1], inds[2]]
        tgt_pts = tgt_pts[mask]
        inds = inds.T[mask]

        if len(gt_boxes) == 0 or len(tgt_pts) == 0:
            tgt_label = torch.zeros_like(tgt_pts[:, 0]).int()
        else:
            boxes = gt_boxes.clone()
            boxes[:, 3] = 0
            pts = pad_r(tgt_pts)
            _, box_idx_of_pts = points_in_boxes_gpu(
                pts, boxes, batch_size=B
            )
            boxes[:, 4:6] *= 4
            _, box_idx_of_pts2 = points_in_boxes_gpu(
                pts, boxes, batch_size=B
            )
            tgt_label = - (box_idx_of_pts2 >= 0).int()
            tgt_label[box_idx_of_pts >= 0] = 1

        n_sam = len(gt_boxes) * 50
        mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
        tgt_label = tgt_label > 0
        return tgt_pts[mask], tgt_label[mask], inds[mask].T

    def assign(self, ctr_pts, samples, B, gt_boxes=None, **kwargs) -> dict:
        """
        Assign target.

        :param ctr_pts: center points of bev maps, including indices, metric centers and regression results.
        :param samples: BEV target point samples.
        :param B: batch size.
        :param gt_boxes: ground truth BBoxes.
        :param kwargs: keyword arguments.
        :return: target_dict that contains the static or/and dynamic target points and their corresponding labels.
        """
        lr = self.lidar_range
        if self.tgt_range is not None:
            ctr_pts, samples = self.filter_range(ctr_pts, samples)
            lr = [-self.tgt_range, -self.tgt_range, -3, self.tgt_range, self.tgt_range, 1]
        if self.downsample:
            ctr_pts = self.down_sample_pred_pts(ctr_pts)

        tgt = {}
        if 'reg_static' in ctr_pts:
            tgt['evi_static'] = draw_sample_evis(
                ctr_pts, samples, 'static', self.res[0], self.distr_r, lr, B, self.var0)
            tgt['lbl_static'] = samples[:, -1]
        if 'reg_dynamic' in ctr_pts:
            assert gt_boxes is not None
            tgt_pts, tgt_label, inds = self.sample_dynamic_tgt_pts(ctr_pts, gt_boxes, B)
            tgt['evi_dynamic'] = draw_sample_evis(
                ctr_pts, tgt_pts, 'dynamic', self.res[0], self.distr_r, lr, B, self.var0)
            tgt['lbl_dynamic'] = tgt_label

        #
        # import matplotlib.pyplot as plt
        # from cosense3d.modules.utils.edl_utils import logit_to_edl
        # fig = plt.figure(figsize=(10, 10))
        # coor = ctr_pts['coor']
        # ctr = ctr_pts['ctr']
        # sams = samples[samples[:, 0]==0][:, 1:].cpu().numpy()
        # mask = coor[:, 0] == 0
        # xy = ctr[mask].cpu().numpy()
        # conf, unc = logit_to_edl(ctr_pts['reg_static'][mask, :2])
        # colors = conf[:, 1].detach().cpu().numpy()
        # plt.scatter(xy[:, 0], xy[:, 1], cmap='jet', c=colors, edgecolors=None, marker='.', s=2, vmin=0, vmax=1)
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(figsize=(10, 10))
        # pos = sams[:, -1] == 1
        # plt.scatter(sams[:, 0], sams[:, 1], c='k', facecolors=None, marker='o', s=5)
        # plt.scatter(sams[pos, 0], sams[pos, 1], c='r', facecolors=None, marker='o', s=5)
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(figsize=(10, 10))
        # mask = tgt_pts[:, 0] == 0
        # sams = tgt_pts[mask][:, 1:].cpu().numpy()
        # pos = tgt_label[mask].cpu().numpy() == 1
        # mask = coor[:, 0] == 0
        # xy = ctr[mask].cpu().numpy()
        # conf, unc = logit_to_edl(ctr_pts['reg_dynamic'][mask, :2])
        # colors = conf[:, 1].detach().cpu().numpy()
        # plt.scatter(xy[:, 0], xy[:, 1], cmap='jet', c=colors, edgecolors=None, marker='.', s=2, vmin=0, vmax=1)
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(figsize=(10, 10))
        # plt.scatter(sams[:, 0], sams[:, 1], c='k', facecolors=None, marker='o', s=5)
        # plt.scatter(sams[pos, 0], sams[pos, 1], c='r', facecolors=None, marker='o', s=5)
        # plt.show()
        # plt.close()

        return tgt

    def get_predictions(self, ctr_pts, B, tag, **kwargs):
        """
        Given center points and its corresponding regressions, generate the dense bev semseg maps
        and its uncertainty and observation mask.

        :param ctr_pts: center points of bev maps, including indices, metric centers and regression results.
        :param B: batch size.
        :param tag: tag for regression key "static | dynamic".
        :param kwargs: keyword arguments
        :return:
            - conf: confidence bev map.
            - unc: uncertainty bev map.
            - obs_mask: observation mask of the bev map.
        """
        reg = ctr_pts[f'reg_{tag}'].relu()
        reg_evi = reg[:, :2]
        reg_var = reg[:, 2:].view(-1, 2, 2)
        ctr = ctr_pts['ctr']
        coor = ctr_pts['coor']

        nbrs = self.nbrs.to(reg_evi.device)
        dists = torch.zeros_like(ctr.view(-1, 1, 2)) + nbrs
        vars0 = [self.var0, self.var0]
        probs_weighted = weighted_mahalanobis_dists(reg_evi, reg_var, dists, vars0)
        voxel_new = ctr.view(-1, 1, 2) + nbrs
        # convert metric voxel points to map indices
        x = (torch.floor(voxel_new[..., 0] / self.res[0]) - self.offset_sz_x).long()
        y = (torch.floor(voxel_new[..., 1] / self.res[1]) - self.offset_sz_y).long()
        batch_indices = (torch.ones_like(probs_weighted[:, :, 0]) * coor[:, :1]).long()
        mask = (x >= 0) & (x < self.size_x) & (y >= 0) & (y < self.size_y)
        x, y = x[mask], y[mask]
        batch_indices = batch_indices[mask]

        # copy sparse probs to the dense evidence map
        indices = batch_indices * self.size_x * self.size_y + x * self.size_y + y
        batch_size = coor[:, 0].max().int().item() + 1
        probs_weighted = probs_weighted[mask].view(-1, 2)
        evidence = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                               device=probs_weighted.device).view(-1, 2)
        torch_scatter.scatter(probs_weighted, indices,
                              dim=0, out=evidence, reduce='sum')
        evidence = evidence.view(batch_size, self.size_x, self.size_y, 2)

        # create observation mask
        obs_mask = torch.zeros_like(evidence[..., 0]).view(-1)
        obs = indices.unique().long()
        obs_mask[obs] = 1
        obs_mask = obs_mask.view(batch_size, self.size_x, self.size_y).bool()
        conf, unc = pred_to_conf_unc(evidence)

        # import matplotlib.pyplot as plt
        # plt.imshow(conf[0, :, :, 1].T.detach().cpu().numpy())
        # plt.show()
        # plt.close()
        return {f'conf_map_{tag}': conf, f'unc_map_{tag}': unc, f'obs_mask_{tag}': obs_mask}


class DiscreteBEVAssigner(BaseAssigner):
    def __init__(self,
                 data_info,
                 stride,
                 down_sample=False,
                 annealing_step=None,
                 ):
        super().__init__()
        update_me_essentials(self, data_info, stride)
        self.down_sample = down_sample
        self.annealing_step = annealing_step

    def pts_to_inds(self, samples):
        """Calculate indices of samples in the bev map"""
        ixy = metric2indices(samples[:, :3], self.res).long()
        ixy[:, 1] -= self.offset_sz_x
        ixy[:, 2] -= self.offset_sz_y
        maskx = torch.logical_and(ixy[:, 1] >= 0, ixy[:, 1] < self.size_x)
        masky = torch.logical_and(ixy[:, 2] >= 0, ixy[:, 2] < self.size_y)
        mask = torch.logical_and(maskx, masky)
        indices = ixy[mask]
        return indices.T, mask

    def get_obs_mask(self, inds, B):
        obs_mask = torch.zeros((B, self.size_x, self.size_y), device=inds.device)
        inds = inds.T
        inds[1] -= self.offset_sz_x
        inds[2] -= self.offset_sz_y
        obs_mask[inds[0], inds[1], inds[2]] = 1
        return obs_mask.bool()


    def assign(self, ctr_pts, samples, B, gt_boxes=None, **kwargs):
        bevmap = self.get_predictions(ctr_pts, B)
        inds, mask = self.pts_to_inds(samples)
        labels = samples[mask][:, -1]
        preds = bevmap[inds[0], inds[1], inds[2]]

        # import matplotlib.pyplot as plt
        # img = pred_to_conf_unc(bevmap)[0][..., 1].detach().cpu().numpy()
        # plt.imshow(img[0].T)
        # plt.show()
        # plt.close()
        return preds, labels

    def get_predictions(self, data_dict, B, edl=True, activation='none', **kwargs):
        reg = data_dict['reg']
        inds = data_dict['coor']
        reg_evi = reg.relu()

        bevmap = torch.zeros((B, self.size_x, self.size_y, reg_evi.shape[-1]),
                             device=reg_evi.device)
        inds = inds.T
        inds[1] -= self.offset_sz_x
        inds[2] -= self.offset_sz_y
        # obs_mask = evidence[..., 0].bool()
        # obs_mask[inds[0], inds[1], inds[2]] = True
        bevmap[inds[0], inds[1], inds[2]] = reg_evi
        return bevmap


class RoIBox3DAssigner(BaseAssigner):
    def __init__(self,
                 box_coder,
                 ):
        self.box_coder = build_box_coder(**box_coder)
        self.code_size = self.box_coder.code_size
    
    def assign(self, pred_boxes, gt_boxes, **kwargs):
        tgt_dict = {
            'rois': [],
            'gt_of_rois': [],
            'gt_of_rois_src': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
            'rois_anchor': [],
            'record_len': []
        }

        for rois, gts in zip(pred_boxes, gt_boxes):
            gts[:, -1] *= 1
            ious = boxes_iou3d_gpu(rois, gts)
            max_ious, gt_inds = ious.max(dim=1)
            gt_of_rois = gts[gt_inds]
            rcnn_labels = (max_ious > 0.3).float()
            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]
            # TODO: roi_ry > 0 in pcdet
            roi_ry = rois[:, 6] % (2 * PI)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = rotate_points_along_z_torch(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * PI), rounding_mode='trunc')
                    + 1) * 2 * PI) % (2 * PI)  # 0 ~ 2pi
            opposite_flag = (heading_label > PI * 0.5) & (
                    heading_label < PI * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + PI) % (
                                                   2 * PI)
            flag = heading_label > PI
            heading_label[flag] = heading_label[
                                      flag] - PI * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-PI / 2,
                                        max=PI / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets, _ = self.box_coder.encode(
                rois_anchor, gt_of_rois.view(-1, self.code_size)
            )

            tgt_dict['rois'].append(rois)
            tgt_dict['gt_of_rois'].append(gt_of_rois)
            tgt_dict['gt_of_rois_src'].append(gt_of_rois_src)
            tgt_dict['cls_tgt'].append(rcnn_labels)
            tgt_dict['reg_tgt'].append(reg_targets)
            tgt_dict['iou_tgt'].append(max_ious)
            tgt_dict['rois_anchor'].append(rois_anchor)
            tgt_dict['record_len'].append(rois.shape[0])

        # cat list to tensor
        for k, v in tgt_dict.items():
            if k == 'record_len':
                continue
            tgt_dict[k] = torch.cat(v, dim=0)
        return tgt_dict

    def get_predictions(self, rcnn_cls, rcnn_iou, rcnn_reg, rois):
        rcnn_cls = rcnn_cls.sigmoid().view(-1)
        rcnn_iou = rcnn_iou.view(-1)
        rcnn_score = rcnn_cls * rcnn_iou**4
        rcnn_reg = rcnn_reg.view(-1, 7)

        rois_anchor = rois.clone().detach().view(-1, self.code_size)
        rois_anchor[:, 0:3] = 0
        rois_anchor[:, 6] = 0

        roi_center = rois[:, 0:3]
        roi_ry = rois[:, 6] % (2 * PI)

        boxes_local = self.box_coder.decode(rois_anchor, rcnn_reg)
        # boxes_local = rcnn_reg + rois_anchor
        detections = rotate_points_along_z_torch(
            points=boxes_local.view(-1, 1, boxes_local.shape[-1]), angle=roi_ry.view(-1)
        ).view(-1, boxes_local.shape[-1])
        detections[:, :3] = detections[:, :3] + roi_center
        detections[:, 6] = detections[:, 6] + roi_ry
        mask = rcnn_score >= 0.01
        detections = detections[mask]
        scores = rcnn_score[mask]

        return {
            'box': detections,
            'scr': scores,
            # Todo currently only support cars
            'lbl': torch.zeros_like(scores),
            # map indices to be aligned with sparse detection head format
            'idx': torch.zeros_like(scores),
        }


class RoadLineAssigner(BaseAssigner):
    def __init__(self,
                 res,
                 range,
                 pos_neg_ratio=2):
        super().__init__()
        self.res = res
        self.range = range
        self.size = int(round(range / res * 2))
        self.pos_neg_ratio = pos_neg_ratio

    def assign(self, coor, tgt_pts, B, **kwargs):
        ctr_coor = coor.clone()
        ctr_coor[:, 1:] = ctr_coor[:, 1:] + self.size / 2
        ctr_coor = ctr_coor.long()
        roadline_maps = torch.zeros((B, self.size, self.size), device=tgt_pts.device)
        mask = (tgt_pts[:, 1:3].abs() < self.range).all(dim=-1)
        tgt_pts = tgt_pts[mask]

        tgt_coor = torch.floor((tgt_pts[:, 1:3] + self.range) / self.res).long()
        mask = torch.logical_and((tgt_coor >= 0).all(dim=-1), (tgt_coor < self.size).all(dim=-1))
        roadline_maps[tgt_pts[mask, 0].long(), tgt_coor[mask, 0], tgt_coor[mask, 1]] = tgt_pts[mask, -1]

        valid = torch.logical_and((ctr_coor[:, 1:3] >= 0).all(dim=-1), (ctr_coor[:, 1:3] < self.size).all(dim=-1))
        labels = roadline_maps[ctr_coor[valid, 0], ctr_coor[valid, 1], ctr_coor[valid, 2]]

        if self.pos_neg_ratio:
            labels = pos_neg_sampling(labels, self.pos_neg_ratio)

        # import matplotlib.pyplot as plt
        # pts_vis = ctr_coor[ctr_coor[:, 0] == 0, 1:].detach().cpu().numpy()
        # lbl_vis = labels.detach().cpu().numpy()
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot()
        # ax.scatter(pts_vis[:, 0], pts_vis[:, 1], c=lbl_vis, marker='.')
        # plt.show()
        # plt.close()
        return labels, valid






