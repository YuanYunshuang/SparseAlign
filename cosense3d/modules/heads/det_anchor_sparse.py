from typing import List

import torch
from torch import nn
from cosense3d.modules import BaseModule
from cosense3d.modules import plugin
from cosense3d.modules.losses import build_loss
from cosense3d.utils.misc import multi_apply
from cosense3d.modules.utils.common import linear_last


class DetAnchorSparse(BaseModule):
    def __init__(self,
                 in_channels,
                 loss_cls,
                 loss_box,
                 num_classes=1,
                 target_assigner=None,
                 get_boxes_when_training=False,
                 get_roi_scores=False,
                 **kwargs):
        super(DetAnchorSparse, self).__init__(**kwargs)
        assert num_classes == 1, 'currently only support binary classification.'
        self.num_classes = num_classes
        self.get_boxes_when_training = get_boxes_when_training
        self.get_roi_scores = get_roi_scores
        self.target_assigner = plugin.build_plugin_module(target_assigner)
        self.num_anchors = self.target_assigner.num_anchors
        self.code_size = self.target_assigner.box_coder.code_size
        self.cls_head = linear_last(in_channels, in_channels * 3, self.num_anchors)
        self.reg_head = linear_last(in_channels, in_channels * 3, self.code_size * self.num_anchors)
        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, stensor_list, **kwargs):
        coor, feat, ctr = self.compose_stensor(stensor_list, self.target_assigner.stride)
        cls = self.cls_head(feat)
        reg = self.reg_head(feat)

        out = {'cls': cls, 'reg': reg, 'ctr': ctr}

        if self.get_roi_scores:
            out['scr'] = cls.sigmoid().max(dim=-1).values

        if self.get_boxes_when_training or not self.training:
            out['preds'] = self.predictions(coor, out)

        return self.format(out, coor, len(stensor_list))

    def format(self, output, coor, B):
        res_list = []
        for i in range(B):
            mask = coor[:, 0] == i
            res_dict = {k: v[mask] for k, v in output.items() if k!='preds'}
            if 'preds' in output:
                preds = {}
                mask = output['preds']['idx'] == i
                for k, v in output['preds'].items():
                    preds[k] = v[mask]
                res_dict['preds'] = preds
            res_list.append(res_dict)
        output = {self.scatter_keys[0]: res_list}
        return output

    def loss(self, preds, gt_boxes, gt_labels, **kwargs):
        pred_cls = self.cat_data_from_list(preds, 'cls')
        pred_reg = self.cat_data_from_list(preds, 'reg')
        coor = [x['ctr'] for x in preds]

        pred_cls = pred_cls.reshape(-1, self.num_classes)
        pred_reg = pred_reg.reshape(-1, self.code_size)
        cls_tgt, reg_tgt, _ = multi_apply(
            self.target_assigner.assign, coor, gt_boxes)
        cls_tgt = torch.cat(cls_tgt, dim=0)
        reg_tgt = torch.cat(reg_tgt, dim=0)[:, :7]

        # vis_cls_pred = pred_cls.view(b, w, h, c).softmax(dim=-1).max(dim=-1).values[0]
        # vis_cls_tgt = cls_tgt.view(b, w, h, c).max(dim=-1).values[0]
        # img = torch.cat([vis_cls_pred, vis_cls_tgt], dim=1).detach().cpu().numpy().T
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(img)
        # plt.show()
        # plt.close()

        pos_mask = cls_tgt > 0
        cared = cls_tgt >= 0
        avg_factor = max(pos_mask.sum(), 1)
        # downsample negative
        # neg_inds = torch.where(cls_tgt == 0)[0]
        # neg_inds = neg_inds[torch.randperm(len(neg_inds))[:avg_factor * 5]]
        # cared[neg_inds] = True

        # focal loss encode the last dim of tgt as background
        labels = pos_mask.new_full((len(pos_mask), ), self.num_classes, dtype=torch.long)
        labels[pos_mask] = 0

        if len(cared) != len(pred_cls):
            print([x['cls'].shape for x in preds])
            print(cared.shape)
        loss_cls = self.loss_cls(pred_cls[cared], labels[cared],
                                 avg_factor=avg_factor)

        reg_preds_sin, reg_tgts_sin = self.add_sin_difference(pred_reg[pos_mask], reg_tgt)
        loss_box = self.loss_box(reg_preds_sin, reg_tgts_sin,
                                 avg_factor=avg_factor / reg_preds_sin.shape[-1])

        return {
            'cls_loss': loss_cls,
            'box_loss': loss_box
        }

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    def predictions(self, coors, preds):
        return self.target_assigner.get_predictions(coors, preds)





