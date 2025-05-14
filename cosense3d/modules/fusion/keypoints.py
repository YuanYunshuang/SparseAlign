import torch
import torch.nn as nn

from cosense3d.modules import BaseModule
from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu
pi = 3.141592653


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class KeypointsFusion(BaseModule):
    def __init__(self, lidar_range, train_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.lidar_range = lidar_range
        self.train_from_epoch = train_from_epoch

    def forward(self, ego_feats, coop_feats, **kwargs):
        epoch = kwargs.get('epoch', self.train_from_epoch + 1)
        if epoch < self.train_from_epoch:
            return {self.scatter_keys[0]: [None for _ in ego_feats]}
        out_dict = {'boxes': [], 'scores': [], 'feat': [], 'coor': []}
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            feat = [ego_feat['point_features']]
            coor = [ego_feat['point_coords']]
            boxes = [ego_feat['boxes']]
            scores = [ego_feat['scores']]
            for cpfeat in coop_feat.values():
                if 'keypoint_feat' not in cpfeat:
                    continue
                feat.append(cpfeat['keypoint_feat']['point_features'])
                coor.append(cpfeat['keypoint_feat']['point_coords'])
                boxes.append(cpfeat['keypoint_feat']['boxes'])
                scores.append(cpfeat['keypoint_feat']['scores'])
            clusters_boxes, clusters_scores = self.clustering(boxes, scores)
            boxes_fused, scores_fused = self.cluster_fusion(clusters_boxes, clusters_scores)
            out_dict['boxes'].append(boxes_fused)
            out_dict['scores'].append(scores_fused)
            out_dict['feat'].append(torch.cat(feat, dim=0))
            out_dict['coor'].append(torch.cat(coor, dim=0))

        return {self.scatter_keys[0]: self.compose_result_list(out_dict, len(ego_feats))}

    def clustering(self, boxes, scores):
        pred_boxes_cat = torch.cat(boxes, dim=0)
        pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
        pred_scores_cat = torch.cat(scores, dim=0)

        if len(pred_scores_cat) == 0:
            clusters = [torch.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.57]).
                                  to(boxes[0].device).view(1, 7)]
            scores= [torch.Tensor([0.01]).to(boxes[0].device).view(-1)]
            return clusters, scores

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
        for j in range(1, cur_cluster_id):
            clusters.append(pred_boxes_cat[cluster_indices == j])
            scores.append(pred_scores_cat[cluster_indices == j])

        return clusters, scores

    @torch.no_grad()
    def cluster_fusion(self, clusters, scores):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        for i, (c, s) in enumerate(zip(clusters, scores)):
            assert len(c) == len(s)
            if len(c) == 1:
                continue
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
            clusters[i] = torch.cat([center_dim.sum(dim=0), theta]).unsqueeze(0)
            s_sorted = torch.sort(s, descending=True).values
            s_fused = 0
            for j, ss in enumerate(s_sorted):
                s_fused += ss ** (j + 1)
            s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
            scores[i] = s_fused

        return torch.cat(clusters, dim=0), torch.cat(scores)


