import torch
from torch import nn

from cosense3d.modules import BaseModule, plugin


class KeypointComposer(BaseModule):
    def __init__(self, vsa, train_from_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.train_from_epoch = train_from_epoch
        self.vsa = plugin.build_plugin_module(vsa)

    def forward(self, preds, bev_feat, voxel_feat, points, **kwargs):
        epoch = kwargs.get('epoch', self.train_from_epoch + 1)
        if epoch < self.train_from_epoch:
            return {self.scatter_keys[0]: [None for _ in preds]}

        res = self.vsa(preds, bev_feat, voxel_feat, points)
        res = self.compose_result_list(res, len(preds))
        return {self.scatter_keys[0]: res}

