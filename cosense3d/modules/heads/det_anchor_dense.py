import os
from typing import List

import torch
from torch import nn
from cosense3d.modules import BaseModule
from cosense3d.modules import plugin
from cosense3d.modules.losses import build_loss
from cosense3d.utils.misc import multi_apply


class DetAnchorDense(BaseModule):
    def __init__(self,
                 in_channels,
                 loss_cls,
                 loss_box,
                 num_classes=1,
                 stride=None,
                 target_assigner=None,
                 get_boxes_when_training=False,
                 box_stamper=None,
                 **kwargs):
        super(DetAnchorDense, self).__init__(**kwargs)
        assert num_classes == 1, 'currently only support binary classification.'
        self.num_classes = num_classes
        self.get_boxes_when_training = get_boxes_when_training
        self.target_assigner = plugin.build_plugin_module(target_assigner)
        self.stride = stride
        if self.stride is None:
            assert target_assigner is not None
            self.stride = self.target_assigner.stride
        self.num_anchors = self.target_assigner.num_anchors
        self.code_size = self.target_assigner.box_coder.code_size
        self.cls_head = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, self.code_size * self.num_anchors, kernel_size=1)
        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)
        if box_stamper is not None:
            self.box_stamper = plugin.build_plugin_module(box_stamper)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, bev_feat_list, points=None, **kwargs):
        if isinstance(bev_feat_list[0], torch.Tensor):
            bev_feat = torch.stack(bev_feat_list, dim=0)
        elif isinstance(bev_feat_list[0], dict):
            bev_feat = torch.stack([x[f'p{self.stride}'] for x in bev_feat_list], dim=0)
        else:
            raise NotImplementedError

        cls = self.cls_head(bev_feat)
        reg = self.reg_head(bev_feat)

        out = {'cls': cls, 'reg': reg}

        if self.get_boxes_when_training or not self.training:
            preds = self.predictions(out)
            if hasattr(self, 'box_stamper'):
                assert points is not None
                preds = self.box_stamper(preds, points)
            out['preds'] = preds

        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        # pos = out['preds']['scr'] > 0.3
        # boxes = out['preds']['box'][pos]
        # inds = out['preds']['idx'][pos]
        # for i in range(2):
        #     pts = points[i][:, :3].detach().cpu().numpy()
        #     box = boxes[inds == i].detach().cpu().numpy()
        #     ax = draw_points_boxes_plt(
        #         pc_range=100,
        #         points=pts,
        #         boxes_pred=box,
        #         return_ax=True
        #     )
        #     plt.savefig(f"{os.environ['HOME']}/Downloads/tmp{i}.jpg")
        #     plt.close()

        return self.format_output(out, len(bev_feat))

    def format_output(self, output, B):
        # decompose batch
        if 'preds' in output:
            preds_list = []
            for i in range(B):
                preds = {}
                mask = output['preds']['idx'] == i
                for k, v in output['preds'].items():
                    preds[k] = v[mask]
                preds_list.append(preds)
            output['preds'] = preds_list
        output = {self.scatter_keys[0]: self.compose_result_list(output, B)}
        return output

    def loss(self, preds, gt_boxes, gt_labels, **kwargs):
        """The dense bev maps show have the shape ((b, c, h, w))"""
        pred_cls = self.stack_data_from_list(preds, 'cls')
        pred_reg = self.stack_data_from_list(preds, 'reg')
        # convert to shape(b, c, h, w) -> (nwh, c) to match the anchors
        b, c, h, w = pred_cls.shape
        pred_cls = pred_cls.permute(0, 3, 2, 1).reshape(-1)
        pred_reg = pred_reg.permute(0, 3, 2, 1).reshape(-1, 7)
        cls_tgt, reg_tgt, _ = multi_apply(
            self.target_assigner.assign, gt_boxes)
        cls_tgt = torch.cat(cls_tgt, dim=0)
        reg_tgt = torch.cat(reg_tgt, dim=0)

        # vis_cls_pred = pred_cls.view(b, w, h, c).sigmoid().max(dim=-1).values[0]
        # topk = vis_cls_pred.view(-1).topk(k=1024).indices
        # wi = topk // w
        # hi = topk % w
        # vis_cls_topk = torch.zeros_like(vis_cls_pred)
        # vis_cls_topk[wi, hi] = 1
        # vis_cls_tgt = cls_tgt.view(b, w, h, c).max(dim=-1).values[0]
        # img = torch.cat([vis_cls_pred, vis_cls_tgt, vis_cls_topk], dim=0).detach().cpu().numpy().T
        # import matplotlib.pyplot as plt
        #
        # plt.imshow(img)
        # plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
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

        loss_cls = self.loss_cls(pred_cls[cared].view(-1, 1), labels[cared],
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

    def predictions(self, preds):
        return self.target_assigner.get_predictions(preds)





