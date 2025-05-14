import copy

import torch.nn as nn
import torch
import numpy as np
from cosense3d.ops import pointnet2_utils
from cosense3d.utils.pclib import rotate_points_along_z_torch
from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu
from cosense3d.utils import box_utils
from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.losses.common import (weighted_smooth_l1_loss,
                                             weighted_sigmoid_binary_cross_entropy)


class KeypointRoIHead(BaseModule):
    def __init__(self,
                 num_cls,
                 in_channels, 
                 n_fc_channels,
                 roi_grid_pool,
                 target_assigner,
                 dp_ratio=0.3,
                 train_from_epoch=0,
                 **kwargs):
        super().__init__(**kwargs)
        self.code_size = 7
        self.dp_ratio = dp_ratio
        self.train_from_epoch = train_from_epoch
        self.target_assigner = plugin.build_plugin_module(target_assigner)
        mlps = copy.copy(roi_grid_pool['mlps'])
        for k in range(len(mlps)):
            mlps[k] = [in_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_utils.StackSAModuleMSG(
            radii=roi_grid_pool['pool_radius'],
            nsamples=roi_grid_pool['n_sample'],
            mlps=mlps,
            use_xyz=True,
            pool_method=roi_grid_pool['pool_method'],
        )

        grid_size = roi_grid_pool['grid_size']
        self.grid_size = grid_size
        c_out = sum([x[-1] for x in mlps])
        pre_channel = grid_size * grid_size * grid_size * c_out
        fc_layers = [n_fc_channels] * 2
        self.shared_fc_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                                  fc_layers)

        self.cls_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                            fc_layers,
                                                            output_channels=
                                                            num_cls)
        self.iou_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                  num_cls)
        self.reg_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=num_cls * 7)

        self._init_weights(weight_init='xavier')

    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def _make_fc_layers(self, input_channels, fc_list, output_channels=None):
        fc_layers = []
        pre_channel = input_channels
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                # nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.dp_ratio > 0:
                fc_layers.append(nn.Dropout(self.dp_ratio))
        if output_channels is not None:
            fc_layers.append(
                nn.Conv1d(pre_channel, output_channels, kernel_size=1,
                          bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers, pre_channel

    def get_global_grid_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        # (B, 6x6x6, 3)
        local_roi_grid_points = self.get_dense_grid_points(rois,
                                                           batch_size_rcnn,
                                                           self.grid_size)
        global_roi_grid_points = rotate_points_along_z_torch(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        """
        Get the local coordinates of each grid point of a roi in the coordinate
        system of the roi(origin lies in the center of this roi.
        """
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = torch.stack(torch.where(faked_features),
                                dim=1)  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1,
                                     1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (
                                  dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(
            dim=1) \
                          - (local_roi_size.unsqueeze(
            dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def roi_grid_pool(self, preds):
        B = len(preds)
        rois = torch.cat([p['boxes'] for p in preds], dim=0)
        point_features = torch.cat([p['feat'] for p in preds], dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)

        xyz = torch.cat([p['coor'] for p in preds], dim=0)
        xyz_batch_cnt = xyz.new_zeros(B).int()
        for bs_idx in range(B):
            xyz_batch_cnt[bs_idx] = len(preds[bs_idx]['coor'])
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(B).int()
        for bs_idx in range(B):
            new_xyz_batch_cnt[bs_idx] = len(preds[bs_idx]['boxes']) * self.grid_size ** 3

        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz[:, :3].contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz[:, :3].contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),  # weighted point features
        )  # (M1 + M2 ..., C)
        # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(-1, self.grid_size ** 3,
                                               pooled_features.shape[-1])

        return pooled_features

    def forward(self, preds, **kwargs):
        epoch = kwargs.get('epoch', self.train_from_epoch + 1)
        if epoch < self.train_from_epoch:
            return {self.scatter_keys[0]: [None for _ in preds]}
        # RoI aware pooling
        pooled_features = self.roi_grid_pool(preds)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, self.grid_size,
                              self.grid_size,
                              self.grid_size)  # (BxN, C, 6, 6, 6)
        shared_features = self.shared_fc_layers(
            pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(
            1, 2).contiguous().squeeze( dim=1)  # (B, 1 or 2)
        rcnn_iou = self.iou_layers(shared_features).transpose(
            1, 2).contiguous().squeeze( dim=1)  # (B, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(
            1, 2).contiguous().squeeze( dim=1)  # (B, C)

        roi_preds = None
        if not self.training:
            rois = torch.cat([p['boxes'] for p in preds], dim=0)
            roi_preds = self.target_assigner.get_predictions(
                rcnn_cls, rcnn_iou, rcnn_reg, rois
            )

        idx = 0
        out_list = []
        for p in preds:
            num = len(p['boxes'])
            out_dict = {
            'rois': p['boxes'],
            'rcnn_cls': rcnn_cls[idx:idx+num],
            'rcnn_iou': rcnn_iou[idx:idx+num],
            'rcnn_reg': rcnn_reg[idx:idx+num],
            }
            if roi_preds is not None:
                out_dict['preds'] = {k: v[idx:idx+num] for k, v in roi_preds.items()}
            out_list.append(out_dict)
            idx += num

        return {self.scatter_keys[0]: out_list}

    def loss(self, out, gt_boxes, epoch, **kwargs):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        if epoch < self.train_from_epoch:
            return {}
        rois = [x['rois'] for x in out]
        label_dict = self.target_assigner.assign(rois, gt_boxes)

        # rcnn out
        rcnn_cls = self.cat_data_from_list(out, 'rcnn_cls').view(1, -1, 1)
        rcnn_iou = self.cat_data_from_list(out, 'rcnn_iou').view(1, -1, 1)
        rcnn_reg = self.cat_data_from_list(out, 'rcnn_reg').view(1, -1, 7)

        tgt_cls = label_dict['cls_tgt'].view(1, -1, 1)
        tgt_iou = label_dict['iou_tgt'].view(1, -1, 1)
        tgt_reg = label_dict['reg_tgt'].view(1, -1, 7)

        pos_norm = tgt_cls.sum()
        # cls loss
        loss_cls = weighted_sigmoid_binary_cross_entropy(rcnn_cls, tgt_cls)

        # iou loss
        # TODO: also count the negative samples
        loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
                                           weights=tgt_cls).mean()

        # regression loss
        # Target resampling : Generate a weights mask to force the regressor concentrate on low iou predictions
        # sample 50% with iou>0.7 and 50% < 0.7
        weights = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        weights[tgt_cls == 0] = 0
        neg = torch.logical_and(tgt_iou < 0.7, tgt_cls != 0)
        pos = torch.logical_and(tgt_iou >= 0.7, tgt_cls != 0)
        num_neg = int(neg.sum(dim=1))
        num_pos = int(pos.sum(dim=1))
        num_pos_smps = max(num_neg, 2)
        pos_indices = torch.where(pos)[1]
        not_selsected = torch.randperm(num_pos)[:num_pos - num_pos_smps]
        # not_selsected_indices = pos_indices[not_selsected]
        weights[:, pos_indices[not_selsected]] = 0
        loss_reg = weighted_smooth_l1_loss(rcnn_reg, tgt_reg,
                                           weights=weights / max(weights.sum(),
                                                                 1)).sum()

        loss_dict = {
            'rcnn_cls_loss': loss_cls,
            'rcnn_iou_loss': loss_iou,
            'rcnn_reg_loss': loss_reg,
        }

        return loss_dict
