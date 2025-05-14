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
from cosense3d.ops.iou3d_nms_utils import nms_gpu


class QueryDetHead(BaseModule):
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
                 sin_enc=False,
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
        self.sin_enc = sin_enc

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

            outputs_classes.append(pred_cls)
            outputs_coords.append(pred_reg)

        all_cls_logits = torch.stack(outputs_classes)
        all_bbox_reg = torch.stack(outputs_coords)

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

        # if self.pred_while_training or not self.training and kwargs['seq_idx'] == 3:
        if kwargs['seq_idx'] == 3:
            dets = self.get_predictions(cls_scores, det_boxes, pred_boxes, batch_inds=reference_inds)
            for i, out in enumerate(outs):
                out['preds'] = self.nms(dets[i])

            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            # inds = reference_inds.unique().int().tolist()
            # if len(inds) == 1: flag = True
            # else: flag = False
            # for i in inds:
            #     mask = reference_inds==i
            #     points = reference_points[mask].detach().cpu().numpy()
            #     boxes = dets[i]['box'][:, :7].detach().cpu().numpy()
            #     scores = cls_scores[0][mask]
            #     scores = scores[:, self.num_classes - 1:].squeeze().detach().cpu().numpy()
            #     ax = draw_points_boxes_plt(
            #         pc_range=self.pc_range.tolist(),
            #         # boxes_pred=boxes,
            #         return_ax=True
            #     )
            #     ax.scatter(points[:, 0], points[:, 1], c=scores, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1)
            #     if flag:
            #         plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
            #         plt.close()
            #     else:
            #         plt.savefig(f"{os.environ['HOME']}/Downloads/tmp_{i}.jpg")
            #         plt.close()
            #
            # print('d')

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

        # if kwargs['itr'] % 1 == 0:
        #     # for i in range(min(2, len(ref_pts))):
        #     if len(ref_pts) // len(petr_out) == 1:
        #         flag = True
        #     else:
        #         flag = False
        #     for i in range(0, len(ref_pts) // len(petr_out), len(petr_out)):
        #         from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        #         points = ref_pts[i].detach().cpu().numpy()
        #         boxes = gt_boxes_global[i][:, :7].detach().cpu().numpy()
        #         scores = petr_out[i]['all_cls_scores'][0]
        #         scores = scores[:, self.num_classes - 1:].squeeze().detach().cpu().numpy()
        #         tgt = cls_tgt.squeeze().cpu().numpy()
        #         ax = draw_points_boxes_plt(
        #             pc_range=self.pc_range.tolist(),
        #             # boxes_pred=boxes,
        #             return_ax=True
        #         )
        #         ax.scatter(points[:, 0], points[:, 1], c=scores, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1)
        #         if flag:
        #             plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        #         else:
        #             plt.savefig(f"{os.environ['HOME']}/Downloads/tmp_{i}.jpg")
        #         plt.close()
        #         if i > 1:
        #             break

        # ax = draw_points_boxes_plt(
        #     pc_range=self.pc_range.tolist(),
        #     boxes_gt=boxes,
        #     return_ax=True
        # )
        # points = ref_pts[0][cared].detach().cpu().numpy()
        # ax.scatter(points[:, 0], points[:, 1], c=tgt, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1)
        # plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        # plt.close()

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
                if reg_dim == 0:
                    continue
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

                if reg_name == 'dir' and reg_dim == 1 and self.sin_enc:
                    cur_reg_src, cur_reg_tgt = self.add_sin_difference(cur_reg_src, cur_reg_tgt)
                cur_loss = self.loss_box(cur_reg_src, cur_reg_tgt)

                loss_box = loss_box + cur_loss
                ptr += reg_dim

        return {
            'cls_loss': loss_cls,
            'box_loss': loss_box,
            'cls_max': pred_to_conf_unc(
                cls_src, self.loss_cls.activation, self.is_edl)[0][..., self.num_classes - 1:].max()
        }

    @staticmethod
    def add_sin_difference(dir1, dir2):
        rad_pred_encoding = torch.sin(dir1) * torch.cos(dir2)
        rad_tg_encoding = torch.cos(dir1) * torch.sin(dir2)
        return rad_pred_encoding, rad_tg_encoding

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

    def nms(self, values):
        out = {}
        boxes = values['box']
        scores = values['scr']
        labels = values['lbl']
        indices = values['idx']  # map index for retrieving features
        if len(values['box']) == 0:
            out.update({
                'box': torch.zeros((0, 7), device=boxes.device),
                'scr': torch.zeros((0,), device=scores.device),
                'lbl': torch.zeros((0,), device=labels.device),
                'idx': torch.zeros(indices.shape[0] if isinstance(indices, torch.Tensor) else (0,),
                                   device=indices.device),
            })
            if 'pred' in values:
                out['pred'] = torch.zeros((0, 2, 7), device=boxes.device)
        else:
            keep = nms_gpu(
                boxes[..., :7],
                scores,
                thresh=0.1,
                pre_maxsize=500
            )
            out.update({
                'box': boxes[keep],
                'scr': scores[keep],
                'lbl': labels[keep],
                'idx': indices[keep],
            })
        return out


class QueryDetHeadSecond(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 code_weights,
                 num_classes,
                 target_assigner,
                 loss_cls,
                 loss_box,
                 num_reg_fcs=3,
                 num_pred=2,
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

        self.target_assigner = build_plugin_module(target_assigner)

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

            outputs_classes.append(pred_cls)
            outputs_coords.append(pred_reg)

        all_cls_logits = torch.stack(outputs_classes)
        all_bbox_reg = torch.stack(outputs_coords)

        det_boxes = self.get_pred_boxes(all_bbox_reg, reference_points)
        cls_scores = pred_to_conf_unc(all_cls_logits.view(1, -1, 1), self.loss_cls.activation, self.is_edl)[0]
        cls_scores = cls_scores.view(1, -1, 2)
        cls_scores, max_inds = cls_scores.max(dim=-1)
        max_inds = max_inds.view(-1)
        inds = torch.arange(cls_scores.shape[1])

        # all_cls_logits = all_cls_logits[:, inds, max_inds]
        # all_bbox_reg = all_bbox_reg.view(1, -1, 2, 9)[:, inds, max_inds]
        det_boxes = det_boxes[inds, max_inds]

        outs = []
        for i in range(len(feat_in)):
            mask = reference_inds == i
            outs.append(
                {
                    'all_cls_logits': all_cls_logits[:, mask].reshape(1, -1, 1),
                    'all_bbox_reg': all_bbox_reg[:, mask].reshape(1, -1, 9),
                    'ref_pts': reference_points[mask],
                    'all_cls_scores': cls_scores[:, mask].reshape(1, -1, 1),
                    'all_bbox_preds': det_boxes[mask].reshape(1, -1, 9),
                    'all_bbox_preds_t': None,
                }
            )

        # if self.pred_while_training or not self.training and kwargs['seq_idx'] == 3:
        if kwargs['seq_idx'] == 3:
            dets = self.get_predictions(cls_scores, det_boxes, None, batch_inds=reference_inds)
            for i, out in enumerate(outs):
                out['preds'] = self.nms(dets[i])

            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            # inds = reference_inds.unique().int().tolist()
            # if len(inds) == 1: flag = True
            # else: flag = False
            # for i in inds:
            #     mask = reference_inds==i
            #     points = reference_points[mask].detach().cpu().numpy()
            #     boxes = dets[i]['box'][:, :7].detach().cpu().numpy()
            #     scores = cls_scores[0][mask]
            #     scores = scores.detach().cpu().numpy()
            #     ax = draw_points_boxes_plt(
            #         pc_range=self.pc_range.tolist(),
            #         # boxes_pred=boxes,
            #         return_ax=True
            #     )
            #     ax.scatter(points[:, 0], points[:, 1], c=scores, cmap='jet', s=3, marker='s', vmin=0.0, vmax=1)
            #     if flag:
            #         plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
            #         plt.close()
            #     else:
            #         plt.savefig(f"{os.environ['HOME']}/Downloads/tmp_{i}.jpg")
            #         plt.close()
            #
            # print('d')

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
                1, self.num_anchors, 1, 1).flatten(0, 1)

        # cls loss
        cls_tgt, reg_tgt, _ = multi_apply(self.target_assigner.assign,
                              ref_pts, gt_boxes_global, **kwargs)
        cls_src = cls_scores.view(-1)

        cls_tgt = torch.cat(cls_tgt, dim=0)
        cared = cls_tgt >= 0
        pos = cls_tgt > 0
        cls_src = cls_src[cared]
        cls_tgt = cls_tgt[cared]

        cur_labels = cls_tgt.long()

        avg_factor = max((cur_labels > 0).sum(), 1)
        loss_cls = self.loss_cls(
            cls_src.view(-1, 1),
            cur_labels,
            temp=epoch,
            avg_factor=avg_factor
        )

        # box loss
        reg_src = bbox_reg[pos]
        reg_tgt = torch.cat(reg_tgt, dim=0)
        reg_preds_sin, reg_tgts_sin = self.add_sin_difference(reg_src, reg_tgt)
        loss_box = self.loss_box(reg_preds_sin, reg_tgts_sin,
                                 avg_factor=avg_factor / reg_preds_sin.shape[-1])



        return {
            'cls_loss': loss_cls,
            'box_loss': loss_box,
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

    def get_pred_boxes(self, bbox_preds, ref_pts):
        reg = {}

        ptr = 0
        for reg_name, reg_dim in self.reg_channels.items():
            reg[reg_name] = bbox_preds[..., ptr:ptr + reg_dim].contiguous()
            ptr += reg_dim

        anchors = self.target_assigner.get_anchors(ref_pts[:, :2])
        box_reg = torch.cat([reg['box'].view(-1, 2, 6),
                             reg['dir'].view(-1, 2, 1),
                             reg['vel'].view(-1, 2, 2)], dim=-1).view(-1, 9)
        box_dec = self.target_assigner.box_coder.decode(anchors, box_reg[:, :7])
        det = torch.cat([box_dec, box_reg[:, 7:]], dim=-1).view(-1, 2, 9)
        return det

    def get_predictions(self, cls_scores, det_boxes, pred_boxes, batch_inds=None):
        cls_scores = cls_scores.view(-1)
        labels = torch.zeros_like(cls_scores)
        pos = cls_scores > self.target_assigner.score_threshold

        dets = []
        if batch_inds is None:
            inds = range(cls_scores.shape[1])
            for i in inds:
                dets.append({
                    'box': det_boxes[-1][i][pos[i]],
                    'scr': cls_scores[i][pos[i]],
                    'lbl': labels[i][pos[i]],
                    'idx': torch.ones_like(labels[i][pos[i]]) * i,
                })
        else:
            inds = batch_inds.unique()
            for i in inds:
                mask = batch_inds == i
                pos_mask = pos[mask]
                dets.append({
                    'box': det_boxes[mask][pos_mask],
                    'scr': cls_scores[mask][pos_mask],
                    'lbl': labels[mask][pos_mask],
                    'pred': pred_boxes[mask][pos_mask] if pred_boxes is not None else None,
                    'idx': batch_inds[mask][pos_mask].long()
                })

        return dets

    def nms(self, values):
        out = {}
        boxes = values['box']
        scores = values['scr']
        labels = values['lbl']
        indices = values['idx']  # map index for retrieving features
        if len(values['box']) == 0:
            out.update({
                'box': torch.zeros((0, 7), device=boxes.device),
                'scr': torch.zeros((0,), device=scores.device),
                'lbl': torch.zeros((0,), device=labels.device),
                'idx': torch.zeros(indices.shape[0] if isinstance(indices, torch.Tensor) else (0,),
                                   device=indices.device),
            })
            if 'pred' in values:
                out['pred'] = torch.zeros((0, 2, 7), device=boxes.device)
        else:
            keep = nms_gpu(
                boxes[..., :7],
                scores,
                thresh=0.1,
                pre_maxsize=500
            )
            out.update({
                'box': boxes[keep],
                'scr': scores[keep],
                'lbl': labels[keep],
                'idx': indices[keep],
            })
        return out


