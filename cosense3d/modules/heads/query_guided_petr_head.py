from typing import List
import os
import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.utils.misc import multi_apply
from cosense3d.utils.box_utils import normalize_bbox, denormalize_bbox
from cosense3d.modules.losses import build_loss
from cosense3d.modules.losses.edl import pred_to_conf_unc


class QueryGuidedPETRHead(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 code_weights,
                 num_classes,
                 cls_assigner,
                 box_assigner,
                 loss_cls,
                 loss_box,
                 num_reg_fcs=3,
                 num_pred=3,
                 use_logits=False,
                 reg_channels=None,
                 sparse=False,
                 pred_while_training=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.reg_channels = {}
        if reg_channels is None:
            self.code_size = 10
        else:
            for c in reg_channels:
                name, channel = c.split(':')
                self.reg_channels[name] = int(channel)
            self.code_size = sum(self.reg_channels.values())
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.num_pred = num_pred
        self.use_logits = use_logits
        self.sparse = sparse
        self.pred_while_training = pred_while_training

        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.code_weights = nn.Parameter(torch.tensor(code_weights), requires_grad=False)

        self.box_assigner = build_plugin_module(box_assigner)
        self.cls_assigner = build_plugin_module(cls_assigner)

        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)
        self.is_edl = True if 'edl' in self.loss_cls.name.lower() else False

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

    def init_weights(self):
        for m in self.cls_branches:
            nn.init.xavier_uniform_(m[-1].weight)
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
        self._is_init = True

    def forward(self, feat_in, **kwargs):
        if self.sparse:
            outs_dec = self.cat_data_from_list(feat_in, 'outs_dec').permute(1, 0, 2)
            reference_points = self.cat_data_from_list(feat_in, 'ref_pts', pad_idx=True)
            reference_inds = reference_points[..., 0]
            reference_points = reference_points[..., 1:]
        else:
            outs_dec = self.stack_data_from_list(feat_in, 'outs_dec').permute(1, 0, 2, 3)
            reference_points = self.stack_data_from_list(feat_in, 'ref_pts')
            reference_inds = None
        pos_dim = reference_points.shape[-1]
        assert outs_dec.isnan().sum() == 0, "found nan in outs_dec."
        # if outs_dec.isnan().any():
        #     print('d')

        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(outs_dec)):
            out_dec = outs_dec[lvl]
            # out_dec = torch.nan_to_num(out_dec)

            pred_cls = self.cls_branches[lvl](out_dec)
            pred_reg = self.reg_branches[lvl](out_dec)

            if self.use_logits:
                reference = inverse_sigmoid(reference_points.clone())
                pred_reg[..., :pos_dim] += reference
                pred_reg[..., :3] = pred_reg[..., :3].sigmoid()

            outputs_classes.append(pred_cls)
            outputs_coords.append(pred_reg)

        all_cls_logits = torch.stack(outputs_classes)
        all_bbox_reg = torch.stack(outputs_coords)
        if self.use_logits:
            all_bbox_reg[..., :3] = (all_bbox_reg[..., :3] * (
                    self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3])

        reference_points = reference_points * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
        det_boxes, pred_boxes = self.get_pred_boxes(all_bbox_reg, reference_points)
        cls_scores = pred_to_conf_unc(all_cls_logits, self.loss_cls.activation, self.is_edl)[0]

        if self.sparse:
            outs = []
            for i in range(len(feat_in)):
                mask = reference_inds == i
                outs.append(
                    {
                        'all_cls_logits': all_cls_logits[:, mask],
                        'all_bbox_reg': all_bbox_reg[:, mask],
                        'ref_pts': reference_points[mask],
                        'all_cls_scores': cls_scores[:, mask],
                        'all_bbox_preds': det_boxes[:, mask],
                        'all_bbox_preds_t': pred_boxes[:, mask] if pred_boxes is not None else None,
                    }
                )
        else:
            outs = [
                {
                    'all_cls_logits': all_cls_logits[:, i],
                    'all_bbox_reg': all_bbox_reg[:, i],
                    'ref_pts': reference_points[i],
                    'all_cls_scores': cls_scores[:, i],
                    'all_bbox_preds': det_boxes[:, i],
                    'all_bbox_preds_t': pred_boxes[:, i] if pred_boxes is not None else None,
                } for i in range(len(feat_in))
            ]

        if self.pred_while_training or not self.training:
            dets = self.get_predictions(cls_scores, det_boxes, pred_boxes, batch_inds=reference_inds)
            for i, out in enumerate(outs):
                out['preds'] = dets[i]

        return {self.scatter_keys[0]: outs}

    def loss(self, petr_out, gt_boxes_global, gt_labels_global, *args, **kwargs):
        aux_dict = {self.gt_keys[2:][i]: x for i, x in enumerate(args)}
        epoch = kwargs.get('epoch', 0)
        if self.sparse:
            cls_scores = torch.cat([x for out in petr_out for x in out['all_cls_logits']], dim=0)
            bbox_reg = torch.cat([x for out in petr_out for x in out['all_bbox_reg']], dim=0)
            ref_pts = [x['ref_pts'] for x in petr_out for _ in range(self.num_pred)]
        else:
            cls_scores = self.stack_data_from_list(petr_out, 'all_cls_logits').flatten(0, 1)
            bbox_reg = self.stack_data_from_list(petr_out, 'all_bbox_reg').flatten(0, 1)
            ref_pts = self.stack_data_from_list(petr_out, 'ref_pts').unsqueeze(1).repeat(
                1, self.num_pred, 1, 1).flatten(0, 1)
        gt_boxes_global = [x for x in gt_boxes_global for _ in range(self.num_pred)]
        # gt_velos = [x[:, 7:] for x in gt_boxes for _ in range(self.num_pred)]
        gt_labels_global = [x for x in gt_labels_global for _ in range(self.num_pred)]
        if 'gt_preds' in aux_dict:
            gt_preds = [x.transpose(1, 0) for x in aux_dict['gt_preds'] for _ in range(self.num_pred)]
        else:
            gt_preds = None

        # cls loss
        cls_tgt = multi_apply(self.cls_assigner.assign,
                              ref_pts, gt_boxes_global, gt_labels_global, **kwargs)
        cls_src = cls_scores.view(-1, self.num_classes)

        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        points = ref_pts[0].detach().cpu().numpy()
        boxes = gt_boxes_global[0][:, :7].detach().cpu().numpy()
        scores = petr_out[0]['all_cls_scores'][0]
        scores = scores[:, self.num_classes - 1:].squeeze().detach().cpu().numpy()
        ax = draw_points_boxes_plt(
            pc_range=self.pc_range.tolist(),
            boxes_gt=boxes,
            return_ax=True
        )
        ax.scatter(points[:, 0], points[:, 1], c=scores, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1)
        plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        plt.close()

        # if kwargs['itr'] % 1 == 0:
        #     from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        #     points = ref_pts[0].detach().cpu().numpy()
        #     boxes = gt_boxes[0][:, :7].detach().cpu().numpy()
        #     scores = pred_to_conf_unc(
        #         cls_scores[0], getattr(self.loss_cls, 'activation'), edl=self.is_edl)[0]
        #     scores = scores[:, self.num_classes - 1:].squeeze().detach().cpu().numpy()
        #     ax = draw_points_boxes_plt(
        #         pc_range=self.pc_range.tolist(),
        #         boxes_gt=boxes,
        #         return_ax=True
        #     )
        #     ax.scatter(points[:, 0], points[:, 1], c=scores, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1.0)
        #     # ax = draw_points_boxes_plt(
        #     #     pc_range=self.pc_range.tolist(),
        #     #     points=points[cls_tgt[0].squeeze().detach().cpu().numpy() > 0],
        #     #     points_c="green",
        #     #     ax=ax,
        #     #     return_ax=True
        #     # )
        #     # ax = draw_points_boxes_plt(
        #     #     pc_range=self.pc_range.tolist(),
        #     #     points=points[scores > 0.5],
        #     #     points_c="magenta",
        #     #     ax=ax,
        #     #     return_ax=True
        #     # )
        #     plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        #     plt.close()

        cls_tgt = torch.cat(cls_tgt, dim=0)
        cared = (cls_tgt >= 0).any(dim=-1)
        cls_src = cls_src[cared]
        cls_tgt = cls_tgt[cared]

        # convert one-hot to labels(
        cur_labels = torch.zeros_like(cls_tgt[..., 0]).long()
        lbl_inds, cls_inds = torch.where(cls_tgt)
        cur_labels[lbl_inds] = cls_inds + 1

        avg_factor = max((cur_labels > 0).sum(), 1)
        loss_cls = self.loss_cls(
            cls_src,
            cur_labels,
            temp=epoch,
            avg_factor=avg_factor
        )

        # box loss
        # pad ref pts with batch index
        if 'gt_preds' in aux_dict:
            gt_preds = self.cat_data_from_list(gt_preds)
        box_tgt = self.box_assigner.assign(
            self.cat_data_from_list(ref_pts, pad_idx=True),
            self.cat_data_from_list(gt_boxes_global, pad_idx=True),
            self.cat_data_from_list(gt_labels_global),
            gt_preds
        )
        ind = box_tgt['idx'][0]  # only one head
        loss_box = 0
        bbox_reg = bbox_reg.view(-1, self.code_size)
        if ind.shape[1] > 0:
            ptr = 0
            for reg_name, reg_dim in self.reg_channels.items():
                pred_reg = bbox_reg[:, ptr:ptr+reg_dim].contiguous()
                if reg_name == 'scr':
                    pred_reg = pred_reg.sigmoid()
                cur_reg_src = pred_reg[box_tgt['valid_mask'][0]]
                if reg_name == 'vel':
                    cur_reg_tgt = box_tgt['vel'][0] * 0.1
                elif reg_name == 'pred':
                    cur_reg_tgt = box_tgt[reg_name][0]
                    mask = cur_reg_tgt[..., 0].bool()
                    cur_reg_src = cur_reg_src[mask]
                    cur_reg_tgt = cur_reg_tgt[mask, 1:]
                else:
                    cur_reg_tgt = box_tgt[reg_name][0]  # N, C
                cur_loss = self.loss_box(cur_reg_src, cur_reg_tgt)

                loss_box = loss_box + cur_loss
                ptr += reg_dim

        return {
            'cls_loss': loss_cls,
            'box_loss': loss_box,
            'cls_max': pred_to_conf_unc(
                cls_src, self.loss_cls.activation, self.is_edl)[0][..., self.num_classes - 1:].max()
        }

    def get_pred_boxes(self, bbox_preds, ref_pts):
        reg = {}

        ptr = 0
        for reg_name, reg_dim in self.reg_channels.items():
            reg[reg_name] = bbox_preds[..., ptr:ptr + reg_dim].contiguous()
            ptr += reg_dim

        out = self.box_assigner.box_coder.decode(ref_pts[None], reg)
        if isinstance(out, tuple):
            det, pred = out
        else:
            det = out
            pred = None
        return det, pred

    def get_predictions(self, cls_scores, det_boxes, pred_boxes, batch_inds=None):
        if self.is_edl:
            scores = cls_scores[-1][..., 1:].sum(dim=-1)
        else:
            scores = cls_scores[-1].sum(dim=-1)
        labels = cls_scores[-1].argmax(dim=-1)
        pos = scores > self.box_assigner.center_threshold

        dets = []
        if batch_inds is None:
            inds = range(cls_scores.shape[1])
            for i in inds:
                dets.append({
                    'box': det_boxes[-1][i][pos[i]],
                    'scr': scores[i][pos[i]],
                    'lbl': labels[i][pos[i]],
                    'idx': torch.ones_like(labels[i][pos[i]]) * i,
                })
        else:
            inds = batch_inds.unique()
            for i in inds:
                mask = batch_inds == i
                pos_mask = pos[mask]
                dets.append({
                    'box': det_boxes[-1][mask][pos_mask],
                    'scr': scores[mask][pos_mask],
                    'lbl': labels[mask][pos_mask],
                    'pred': pred_boxes[-1][mask][pos_mask] if pred_boxes is not None else None,
                    'idx': batch_inds[mask][pos_mask].long()
                })

        return dets


