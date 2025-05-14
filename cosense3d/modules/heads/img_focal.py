import torch
from torch import nn
from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.init import bias_init_with_prob
from cosense3d.modules.utils.common import inverse_sigmoid, clip_sigmoid
from cosense3d.utils.box_utils import bbox_xyxy_to_cxcywh
from cosense3d.utils.iou2d_calculator import bbox_overlaps
from cosense3d.utils.misc import multi_apply
from cosense3d.modules.losses import build_loss


class ImgFocal(BaseModule):
    def __init__(self, in_channels, embed_dims, num_classes, center_assigner, box_assigner,
                 loss_cls2d, loss_centerness, loss_bbox2d, loss_iou2d, loss_centers2d,
                 with_depth=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.with_depth = with_depth

        self.center_assigner = plugin.build_plugin_module(center_assigner)
        self.box_assigner = plugin.build_plugin_module(box_assigner)

        self.loss_cls2d = build_loss(**loss_cls2d)
        self.loss_centerness = build_loss(**loss_centerness)
        self.loss_bbox2d = build_loss(**loss_bbox2d)
        self.loss_iou2d = build_loss(**loss_iou2d)
        self.loss_centers2d = build_loss(**loss_centers2d)

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)
        self.shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)
        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

        self.shared_reg = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=self.embed_dims),
            nn.ReLU(), )
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)
        if self.with_depth:
            self.depth = nn.Conv2d(self.embed_dims, 1, kernel_size=1)

    def forward(self, img_feat, img_coor, **kwargs):
        out_dict = {}
        x = self.cat_data_from_list(img_feat)
        N, c, h, w = x.shape
        n_pixels = h * w

        cls_feat = self.shared_cls(x)
        cls = self.cls(cls_feat)
        centerness = self.centerness(cls_feat)
        cls_logits = cls.permute(0,2,3,1).reshape(-1, n_pixels, self.num_classes)
        centerness = centerness.permute(0,2,3,1).reshape(-1, n_pixels, 1)
        cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(-1, n_pixels, 1)
        sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(-1, n_pixels, 1).sigmoid()

        out_dict.update({
            'feat_size': [h, w],
            'centerness': centerness,
            'cls_score': cls_score,
            'sample_weight': sample_weight
        })


        img_coor = self.cat_data_from_list(img_coor)
        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous()
        ltrb = ltrb.sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()
        centers2d = self.apply_center_offset(img_coor, centers2d_offset)
        bboxes = self.apply_ltrb(img_coor, ltrb)

        pred_bboxes = bboxes.view(-1, n_pixels, 4)
        pred_centers2d = centers2d.view(-1, n_pixels, 2)
        out_dict.update({
            'pred_boxes': pred_bboxes,
            'pred_centers2d': pred_centers2d
        })

        if self.with_depth:
            # TODO
            raise NotImplementedError

        return self.format_output(out_dict, img_feat)

    def format_output(self, out_dict, img_feat):
        ptr = 0
        output_list = []
        for imgs in img_feat:
            n = imgs.shape[0]
            output_list.append({k: v[ptr:ptr+n] for k, v in out_dict.items()})
            ptr += n
        return {self.scatter_keys[0]: output_list}

    def loss(self, batch_list, labels2d, centers2d, bboxes2d, img_size, **kwargs):
        feat_size = batch_list[0]['feat_size']
        centerness = self.cat_data_from_list(batch_list, 'centerness')
        cls_score = self.cat_data_from_list(batch_list, 'cls_score')
        pred_boxes = self.cat_data_from_list(batch_list, 'pred_boxes')
        pred_centers2d = self.cat_data_from_list(batch_list, 'pred_centers2d')
        labels2d = self.cat_list(labels2d)
        centers2d = self.cat_list(centers2d)
        bboxes2d = self.cat_list(bboxes2d)
        img_size = self.cat_list(img_size)
        B = len(img_size)

        num_gts, assigned_gt_inds, assigned_labels = multi_apply(
            self.box_assigner.assign,
            pred_boxes, cls_score, pred_centers2d,
            bboxes2d, labels2d, centers2d, img_size)

        cared_pred_boxes = []
        cared_centers = []
        aligned_bboxes_gt = []
        aligned_centers_gt = []
        aligned_labels = []
        factors = []
        mask = []
        for i, s in enumerate(img_size):
            pos_mask = assigned_gt_inds[i] > 0
            mask.append(pos_mask)
            pos_inds = assigned_gt_inds[i][pos_mask] - 1
            boxes = pred_boxes[i][pos_mask]
            cared_pred_boxes.append(boxes)
            factors.append(pred_boxes.new_tensor(
                [s[1], s[0], s[1], s[0]]).unsqueeze(0).repeat(boxes.shape[0], 1))
            aligned_bboxes_gt.append(bboxes2d[i][pos_inds])
            cared_centers.append(pred_centers2d[i][pos_mask])
            aligned_centers_gt.append(centers2d[i][pos_inds])
            labels = pos_mask.new_full((len(pos_mask), ), self.num_classes, dtype=torch.long)
            labels[pos_mask] = labels2d[i][pos_inds]
            aligned_labels.append(labels)

        factors = torch.cat(factors, dim=0)
        cared_pred_boxes = torch.cat(cared_pred_boxes, dim=0)
        cared_pred_boxes_pix = cared_pred_boxes * factors
        cared_centers = torch.cat(cared_centers, dim=0)
        factors_inv = 1 / factors
        aligned_bboxes_gt = torch.cat(aligned_bboxes_gt, dim=0)
        aligned_centers_gt = torch.cat(aligned_centers_gt, dim=0)
        aligned_labels = torch.cat(aligned_labels, dim=0)
        mask = torch.cat(mask, dim=0)

        loss_iou = self.loss_iou2d(cared_pred_boxes_pix, aligned_bboxes_gt)

        cls_score = cls_score.reshape(-1, cls_score.shape[-1])
        iou_score = torch.zeros_like(cls_score[..., 0])
        iou_score[mask] = bbox_overlaps(aligned_bboxes_gt, cared_pred_boxes_pix,
                                        is_aligned=True).reshape(-1)
        cls_avg_factor = max(sum(num_gts), 1)
        loss_cls = self.loss_cls2d(
            cls_score, (aligned_labels, iou_score.detach()), avg_factor=cls_avg_factor)

        loss_box = self.loss_bbox2d(cared_pred_boxes, aligned_bboxes_gt * factors_inv)
        loss_center = self.loss_centers2d(cared_centers, aligned_centers_gt * factors_inv[:, :2])

        heatmaps = multi_apply(self.center_assigner.assign, centers2d, bboxes2d,
                               img_size, [img_size[0][0] // feat_size[0]] * B)
        heatmaps = torch.stack(heatmaps, dim=0).view(B, -1, 1)
        centerness = clip_sigmoid(centerness).view(B, -1, 1)
        loss_centerness = self.loss_centerness(centerness, heatmaps, avg_factor=cls_avg_factor)
        return {
            'img_cls_loss': loss_cls,
            'img_iou_loss': loss_iou,
            'img_box_loss': loss_box,
            'img_ctr_loss': loss_center,
            'img_ctrness_loss': loss_centerness,
        }


    @staticmethod
    def apply_center_offset(locations, center_offset):
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4)
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

    @staticmethod
    def apply_ltrb(locations, pred_ltrb):
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4)
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])  # x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])  # y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])  # x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])  # y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)

        return pred_boxes


