import torch
import torch.nn as nn

from cosense3d.modules import BaseModule
from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu
pi = 3.141592653


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class BoxFusion(BaseModule):
    def __init__(self, lidar_range, **kwargs):
        super().__init__(**kwargs)
        self.lidar_range = lidar_range

    def forward(self, ego_preds, coop_preds, memory, global_times, **kwargs):
        out_dict = {'box': [], 'scr': [], 'lbl': [], 'time': [], 'idx': []}
        for ego_pred, coop_pred, mem, global_time in zip(ego_preds, coop_preds, memory, global_times):
            boxes = [ego_pred['preds']['box']]
            scores = [ego_pred['preds']['scr']]
            labels = [ego_pred['preds']['lbl']]
            times = [ego_pred['preds']['time']]
            if len(mem) > 0:
                boxes.append(mem['preds']['box'])
                scores.append(mem['preds']['scr'])
                labels.append(mem['preds']['lbl'])
                times.append(mem['preds']['time'])
            for cppred in coop_pred.values():
                boxes.append(cppred['detection_local']['preds']['box'])
                scores.append(cppred['detection_local']['preds']['scr'])
                labels.append(cppred['detection_local']['preds']['lbl'])
                times.append(cppred['detection_local']['preds']['time'])
            clusters_boxes, clusters_scores, cluster_labels, cluster_times = \
                self.clustering(boxes, scores, labels, times, global_time)
            boxes_fused, scores_fused, labels_fused, times_fused = self.cluster_fusion(
                clusters_boxes, clusters_scores, cluster_labels, cluster_times, global_time)
            out_dict['box'].append(boxes_fused)
            out_dict['scr'].append(scores_fused)
            out_dict['lbl'].append(labels_fused)
            out_dict['time'].append(times_fused)
            out_dict['idx'].append(torch.zeros_like(labels_fused))

        out_list = self.compose_result_list(out_dict, len(ego_preds))
        return {self.scatter_keys[0]: [{'preds': x} for x in out_list]}

    def clustering(self, boxes, scores, labels, times, global_time):
        times_cat = torch.cat(times, dim=0)
        # remove boxes outside the maximum time length
        mask = (global_time - times_cat) < 0.15
        pred_boxes_cat = torch.cat(boxes, dim=0)[mask]
        pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
        pred_scores_cat = torch.cat(scores, dim=0)[mask]
        pred_labels_cat = torch.cat(labels, dim=0)[mask]
        times_cat = times_cat[mask]

        if len(pred_scores_cat) == 0:
            clusters = [torch.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.57]).
                                  to(boxes[0].device).view(1, 7)]
            scores= [torch.Tensor([0.01]).to(boxes[0].device).view(-1)]
            labels = [torch.Tensor([-1]).to(boxes[0].device).view(-1)]
            times = [torch.Tensor([-1]).to(boxes[0].device).view(-1)]
            return clusters, scores, labels, times

        ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
        cluster_indices = torch.zeros(len(ious)).int()  # gt assignments of preds
        cur_cluster_id = 1
        while torch.any(cluster_indices == 0):
            cur_idx = torch.where(cluster_indices == 0)[0][
                0]  # find the idx of the first pred which is not assigned yet
            cluster_indices[torch.where(ious[cur_idx] > 0.1)[0]] = cur_cluster_id
            cur_cluster_id += 1
        clusters = []
        scores = []
        labels = []
        times = []
        for j in range(1, cur_cluster_id):
            clusters.append(pred_boxes_cat[cluster_indices == j])
            scores.append(pred_scores_cat[cluster_indices == j])
            labels.append(pred_labels_cat[cluster_indices == j])
            times.append(times_cat[cluster_indices == j])

        return clusters, scores, labels, times

    @torch.no_grad()
    def cluster_fusion(self, clusters, scores, labels, times, global_time):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        for i, (c, s, l, t) in enumerate(zip(clusters, scores, labels, times)):
            assert len(c) == len(s)
            if len(c) == 1:
                labels[i] = l[0]
                times[i] = t[0]
                continue
            uniq_lbls, cnt = l.mode(keepdim=True)
            labels[i] = uniq_lbls[cnt.argmax()]


            box_fused, s_fused = self.merge_sync_boxes(c, s)
            scores[i] = s_fused
            clusters[i] = box_fused
            times[i] = t.mean()

        return torch.cat(clusters, dim=0), torch.cat(scores), torch.stack(labels), torch.stack(times)

    @torch.no_grad()
    def temporal_cluster_fusion(self, clusters, scores, labels, times, global_time):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        for i, (c, s, l, t) in enumerate(zip(clusters, scores, labels, times)):
            assert len(c) == len(s)
            if len(c) == 1:
                labels[i] = l[0]
                times[i] = t[0]
                continue
            uniq_lbls, cnt = l.mode(keepdim=True)
            labels[i] = uniq_lbls[cnt.argmax()]

            t_idx = (t * 100).round().int()
            uniq_ts = torch.unique(t_idx)
            ts = []
            boxes = []
            scrs = []
            for idx in uniq_ts:
                mask = t_idx == idx
                cur_cluster = c[mask]
                cur_scores = s[mask]
                box_fused, s_fused = self.merge_sync_boxes(cur_cluster, cur_scores)
                ts.append(t[mask].mean())
                boxes.append(box_fused)
                scrs.append(s_fused)

            if len(ts) == 1:
                scores[i] = scrs[0]
                clusters[i] = boxes[0]
                times[i] = ts[0]
            else:
                # interpolate to global time
                ts = torch.stack(ts)
                sort_inds = torch.argsort(ts)
                ts = ts[sort_inds]
                boxes = torch.cat(boxes, dim=0)[sort_inds]
                scrs = torch.cat(scrs)[sort_inds]
                velo = (boxes[-1, :2] - boxes[-2, :2]) / (ts[-1] - ts[-2])
                out_box = boxes[scrs.argmax()]
                out_box[:2] += velo * (global_time - ts[-1])
                scores[i] = torch.mean(scrs, dim=0, keepdim=True)
                clusters[i] = out_box.reshape(1, -1)
                times[i] = torch.tensor(global_time, device=ts.device)

        return torch.cat(clusters, dim=0), torch.cat(scores), torch.stack(labels), torch.stack(times)

    def merge_sync_boxes(self, c, s):
        # reverse direction for non-dominant direction of boxes
        dirs = c[:, -1]
        max_score_idx = torch.argmax(s)
        dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
        lt_pi = (dirs_diff > pi).int()
        dirs_diff = dirs_diff * (1 - lt_pi) + (
                2 * pi - dirs_diff) * lt_pi
        score_lt_half_pi = s[dirs_diff > pi / 2].sum()  # larger than
        score_set_half_pi = s[
            dirs_diff <= pi / 2].sum()  # small equal than
        # select larger scored direction as final direction
        if score_lt_half_pi <= score_set_half_pi:
            dirs[dirs_diff > pi / 2] += pi
        else:
            dirs[dirs_diff <= pi / 2] += pi
        dirs = limit_period(dirs)
        s_normalized = s / s.sum()
        sint = torch.sin(dirs) * s_normalized
        cost = torch.cos(dirs) * s_normalized
        theta = torch.atan2(sint.sum(), cost.sum()).view(1, )
        center_dim = c[:, :-1] * s_normalized[:, None]
        box_fused = torch.cat([center_dim.sum(dim=0), theta]).unsqueeze(0)
        s_sorted = torch.sort(s, descending=True).values
        s_fused = 0
        for j, ss in enumerate(s_sorted):
            s_fused += ss ** (j + 1)
        s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
        return box_fused, s_fused