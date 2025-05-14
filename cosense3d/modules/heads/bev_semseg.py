import os

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.common import pad_r, linear_last, cat_coor_with_idx
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses import edl, build_loss
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.plugin.attn import NeighborhoodAttention


class SemsegHead(BaseModule):
    def __init__(self,
                 data_info,
                 in_dim,
                 stride,
                 target_assigner,
                 loss_cls,
                 num_cls=2,
                 static_head=True,
                 dynamic_head=True,
                 **kwargs):
        super(SemsegHead, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.stride = stride
        self.num_cls = num_cls
        for k, v in data_info.items():
            setattr(self, k, v)
        update_me_essentials(self, data_info, self.stride)
        assert (static_head or dynamic_head), "At least one of static_head or dynamic_head should be True."

        self.init_layers(in_dim, num_cls, static_head, dynamic_head)
        self.tgt_assigner = build_plugin_module(target_assigner)
        self.loss_cls = build_loss(**loss_cls)
        self.is_edl = True if 'edl' in self.loss_cls.name.lower() else False

    def init_layers(self, in_dim, num_cls, static_head, dynamic_head):
        raise NotImplementedError

    def forward(self, stensor_list, **kwargs):
        B = len(stensor_list)
        coor, feat, ctr = self.format_input(stensor_list)

        out = {'ctr': ctr, 'coor': coor}
        if hasattr(self, 'static_head'):
            out['reg_static'] = self.static_head(feat)
            if not self.training:
                out.update(self.tgt_assigner.get_predictions(out, B, 'static'))
        if hasattr(self, 'dynamic_head'):
            out['reg_dynamic'] = self.dynamic_head(feat)
            if not self.training:
                out.update(self.tgt_assigner.get_predictions(out, B, 'dynamic'))

        # a1 = (out['reg_static'] > 0).sum(0)
        # a2 = (out['reg_dynamic'] > 0).sum(0)
        # import matplotlib.pyplot as plt
        # from cosense3d.modules.utils.edl_utils import logit_to_edl
        # fig = plt.figure(figsize=(14, 5))
        # mask = coor[:, 0] == 0
        # xy = ctr[mask].detach().cpu().numpy()
        # conf, unc = logit_to_edl(out['reg_static'][mask, :2])
        # colors = conf[:, 1].detach().cpu().numpy()
        # # neg = colors <= 0.5
        # plt.scatter(xy[:, 0], xy[:, 1], cmap='jet', c=colors, edgecolors=None, marker='.', s=1, vmin=0, vmax=1)
        # plt.show()
        # plt.close()

        return self.format_output(out, B)

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        batch_inds = output['coor'][:, 0]
        output['coor'] = output['coor'][:, 1:]

        for i in range(B):
            mask = batch_inds == i
            for k in output.keys():
                if 'map' in k or 'mask' in k:
                    output_new[k].append(output[k][i])
                else:
                    output_new[k].append(output[k][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def loss(self, batch_list, tgt_pts, gt_boxes, **kwargs):
        coor = self.cat_data_from_list(batch_list, 'coor', pad_idx=True)
        coor[:, 1:] = coor[:, 1:] / self.stride
        keys = list(batch_list[0].keys())
        keys.remove('coor')
        ctr_pts = {'coor': coor}
        for k in keys:
            ctr_pts[k] = self.cat_data_from_list(batch_list, k)
        B = len(tgt_pts)
        tgt_pts = cat_coor_with_idx(tgt_pts)
        gt_boxes = cat_coor_with_idx(gt_boxes)

        tgt = self.tgt_assigner.assign(
            ctr_pts, tgt_pts, B, gt_boxes, **kwargs)
        epoch_num = kwargs.get('epoch', 0)

        loss = 0
        loss_dict = {}
        if 'reg_static' in keys:
            loss, loss_dict = self.cal_loss(loss_dict, loss, tgt, 'static', epoch_num)
        if 'reg_dynamic' in keys:
            loss, loss_dict = self.cal_loss(loss_dict, loss, tgt, 'dynamic', epoch_num)

        loss_dict['bev_loss'] = loss
        return loss_dict

    def cal_loss(self, loss_dict, loss, tgt, tag, epoch_num, **kwargs):
        loss_cls = self.loss_cls(
            tgt[f'evi_{tag}'],
            tgt[f'lbl_{tag}'],
            temp=epoch_num,
        )
        loss = loss + loss_cls
        loss_dict[f'bev_{tag}_loss'] = loss_cls
        return loss, loss_dict

    def draw_bev_map(self, data_dict, B, **kwargs):
        return self.tgt_assigner.get_predictions(data_dict, B, **kwargs)


class GevSemsegHead(SemsegHead):
    def init_layers(self, in_dim, num_cls, static_head, dynamic_head):
        if static_head:
            self.static_head = linear_last(in_dim, 32, num_cls * 3, bias=True)
        if dynamic_head:
            self.dynamic_head = linear_last(in_dim, 32, num_cls * 3, bias=True)


class EviSemsegHead(SemsegHead):
    def init_layers(self, in_dim, num_cls, static_head, dynamic_head):
        if static_head:
            self.static_head = linear_last(in_dim, 32, num_cls, bias=True)
        if dynamic_head:
            self.dynamic_head = linear_last(in_dim, 32, num_cls, bias=True)


