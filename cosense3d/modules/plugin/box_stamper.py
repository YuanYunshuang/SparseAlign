import torch
from torch import nn
from torch_scatter import scatter_mean

from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.utils.common import cat_coor_with_idx


class BoxTimeStamper(nn.Module):
    def __init__(self, filter_empty_boxes=True, **kwargs):
        super().__init__()
        self.filter_empty_boxes = filter_empty_boxes

    def forward(self, preds, points_list):
        boxes = torch.cat([preds['idx'].view(-1, 1), preds['box']], dim=-1)
        points = cat_coor_with_idx(points_list)

        box_idx_of_pts = points_in_boxes_gpu(points[:, :4], boxes,
                                             batch_size=len(points_list))[1]
        mask = box_idx_of_pts >= 0
        inds = box_idx_of_pts[mask]
        times = points[mask, -1]
        mean_time = times.new_zeros(boxes.shape[:1])
        scatter_mean(times, inds, dim=0, out=mean_time)
        preds['time'] = mean_time

        valid = inds.unique()
        for k, v in preds.items():
            preds[k] = v[valid]
        return preds