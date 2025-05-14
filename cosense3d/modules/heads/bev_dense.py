"""
Seg head for bev understanding
"""

import torch
import torch.nn as nn
from einops import rearrange

from cosense3d.modules import BaseModule
from cosense3d.modules.losses import build_loss


class BevSegHead(BaseModule):
    def __init__(self, target, input_dim, output_class, loss_cls, **kwargs):
        super(BevSegHead, self).__init__(**kwargs)
        self.target = target
        if 'dynamic' in self.target:
            self.dynamic_head = nn.Conv2d(input_dim,
                                          output_class,
                                          kernel_size=3,
                                          padding=1)
        if 'static' in self.target:
            self.static_head = nn.Conv2d(input_dim,
                                         output_class,
                                         kernel_size=3,
                                         padding=1)
        self.loss_cls = build_loss(**loss_cls)

    def forward(self,  x, **kwargs):
        x = self.stack_data_from_list(x)
        out_dict = {}
        if 'dynamic' in self.target:
            out_dict['dynamic_bev_pred'] = self.dynamic_head(x)
            if not self.training:
                out_dict['dynamic_bev_pred'] = out_dict['dynamic_bev_pred'].permute(0, 2, 3, 1).softmax(dim=-1)
        if 'static' in self.target:
            out_dict['dynamic_bev_pred'] = self.static_head(x)
            if not self.training:
                out_dict['static_bev_pred'] = out_dict['dynamic_bev_pred'].permute(0, 2, 3, 1).softmax(dim=1)

        # output_list = self.compose_result_list(out_dict, len(x))
        return out_dict

    def loss(self, dynamic_bev_preds, dynamic_bev, **kwargs):
        dynamic_bev_preds = self.stack_data_from_list(dynamic_bev_preds)
        dynamic_bev_gt = torch.stack(dynamic_bev, dim=0)
        loss_dict = self.loss_cls(
            dynamic_pred=dynamic_bev_preds,
            dynamic_gt=dynamic_bev_gt
        )
        return loss_dict


class BevRoIDenseHead(BaseModule):
    def __init__(self, in_dim, stride, num_cls=1, loss_cls=None, **kwargs):
        super(BevRoIDenseHead, self).__init__(**kwargs)
        self.head = nn.Conv2d(in_dim, num_cls, kernel_size=1)
        self.stride = stride
        if loss_cls is not None:
            self.loss_cls = build_loss(**loss_cls)

    def forward(self,  input, **kwargs):
        x = self.stack_data_from_list([x[f'p{self.stride}'] for x in input])
        x = self.head(x)

        # output_list = self.compose_result_list(out_dict, len(x))
        return {self.scatter_keys[0]: x}

    def loss(self, bev_preds, bev_tgt, **kwargs):
        bev_preds = self.stack_data_from_list(bev_preds)
        dynamic_bev_gt = torch.stack(bev_tgt, dim=0)
        loss_dict = self.loss_cls(
            dynamic_pred=bev_preds,
            dynamic_gt=dynamic_bev_gt
        )
        return loss_dict


