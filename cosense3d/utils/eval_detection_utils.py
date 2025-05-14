import os

import numpy as np
import torch

from cosense3d.utils.misc import torch_tensor_to_numpy
from cosense3d.utils.box_utils import convert_box_to_polygon, compute_iou, boxes_to_corners_3d
from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev, boxes_iou3d_cpu, boxes_bev_iou_cpu


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    det_range=None, return_ious=False):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2) or (N, 7).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    range : list, [left_range, right_range]
        The evaluation range left bound
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    det_gt_ious = None
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = torch_tensor_to_numpy(det_boxes)
        det_score = torch_tensor_to_numpy(det_score)
        gt_boxes = torch_tensor_to_numpy(gt_boxes)
        det_gt_ious = np.zeros_like(det_boxes[:, 0])
        # convert center format to corners
        if det_boxes.ndim==2 and det_boxes.shape[1] == 7:
            det_boxes = boxes_to_corners_3d(det_boxes)
        if gt_boxes.ndim==2 and gt_boxes.shape[1] == 7:
            gt_boxes = boxes_to_corners_3d(gt_boxes)

        # remove the bbx out of range
        if det_range is not None:
            pass

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend]  # from high to low
        det_polygon_list = list(convert_box_to_polygon(det_boxes))
        gt_polygon_list = list(convert_box_to_polygon(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = compute_iou(det_polygon, gt_polygon_list)
            if len(gt_polygon_list) > 0 and return_ious:
                det_gt_ious[score_order_descend[i]] = np.max(ious)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
        result_stat[iou_thresh]['scr'] += det_score.tolist()
    else:
        gt = gt_boxes.shape[0]
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt

    if return_ious:
        return det_gt_ious


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.

    iou : float

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['scr'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()

    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, iou_thrs, global_sort_detections=False):
    dump_dict = {}
    for iou in iou_thrs:
        ap, mrec, mpre = calculate_ap(result_stat, iou, global_sort_detections)
        iou_str = f"{int(iou * 100)}"
        dump_dict.update({f'ap_{iou_str}': ap,
                          f'mpre_{iou_str}': mpre,
                          f'mrec_{iou_str}': mrec,
                          })
    return dump_dict


def ops_cal_tp(pred_boxes, gt_boxes, iou_mode='3d', IoU_thr=0.7):
    if len(pred_boxes) == 0:
        return torch.zeros(pred_boxes.shape[0], device=pred_boxes.device)
    elif len(gt_boxes) == 0:
        return torch.zeros(len(pred_boxes), device=pred_boxes.device).bool()
    else:
        if pred_boxes.is_cuda:
            iou_func = boxes_iou3d_gpu if iou_mode == '3d' else boxes_iou_bev
        else:
            iou_func = boxes_iou3d_cpu if iou_mode == '3d' else boxes_bev_iou_cpu
        ious = iou_func(pred_boxes, gt_boxes)
        max_iou_pred_to_gts = ious.max(dim=1)
        max_iou_gt_to_preds = ious.max(dim=0)
        tp = max_iou_pred_to_gts[0] > IoU_thr
        is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                        == torch.tensor([i for i in range(len(tp))], device=tp.device)
        tp[torch.logical_not(is_best_match)] = False
        return tp


def cal_precision_recall(scores, tps, n_pred, n_gt):
    order_inds = scores.argsort(descending=True)
    tp_all = tps[order_inds]
    list_accTP = tp_all.cumsum(dim=0)
    precision = list_accTP.float() / torch.arange(1, n_pred + 1)
    recall = list_accTP.float() / n_gt
    return precision, recall


def cal_ap_all_point(scores, tps, n_pred, n_gt):
    '''
    source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L292
    '''

    prec, rec = cal_precision_recall(scores, tps, n_pred, n_gt)
    mrec = []
    mrec.append(0)
    [mrec.append(e.item()) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e.item()) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]