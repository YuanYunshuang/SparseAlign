from functools import partial
from typing import List

import spconv
import torch
import torch.nn as nn

from spconv.pytorch import  SparseSequential, SubMConv3d, SparseConv3d, SparseInverseConv3d, SparseConvTensor
from cosense3d.modules import BaseModule, plugin


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class Spconv(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 voxel_generator,
                 voxel_encoder,
                 bev_neck=None,
                 bev_compressor=None,
                 cache_coords=True,
                 cache_strides=[1, 2, 4, 8],
                 **kwargs):
        super(Spconv, self).__init__(**kwargs)
        self.num_point_features = out_channels
        self.cache_keys = []
        if cache_coords:
            self.cache_keys.append('coords')
        for s in cache_strides:
            self.cache_keys.append(f'p{s}')
        self.voxel_generator = plugin.build_plugin_module(voxel_generator)
        self.voxel_encoder = plugin.build_plugin_module(voxel_encoder)
        self.grid_size = self.voxel_generator.grid_size
        if bev_neck is not None:
            self.bev_neck = plugin.build_plugin_module(bev_neck)
        if bev_compressor is not None:
            self.bev_compressor = plugin.build_plugin_module(bev_compressor)
        self._init_layers(in_channels, out_channels)

    def _init_layers(self, in_channels, out_channels):
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = self.grid_size.tolist()[::-1]
        self.sparse_shape[0] += 1

        self.conv_input = SparseSequential(
            SubMConv3d(in_channels, 16, 3,
                       padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = SparseSequential(
            block(16, 16, 3,
                  norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3,
                  norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3,
                  norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3,
                  norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            SparseConv3d(64, out_channels, (3, 1, 1),
                         stride=(2, 1, 1), padding=last_pad, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channels),
            nn.ReLU(),
        )

        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64,
            'out': out_channels
        }

    def forward(self, points: list, **kwargs):
        B = len(points)
        res_dict = {}
        voxels, coords, num_points = self.voxel_generator(x[:, :4] for x in points)
        res_dict['coords'] = coords
        coords = self.cat_data_from_list(coords, pad_idx=True)
        voxels = self.cat_data_from_list(voxels)
        num_points = self.cat_data_from_list(num_points)
        voxel_features = self.voxel_encoder(voxels, num_points)

        input_sp_tensor = SparseConvTensor(
            features=voxel_features,
            indices=coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=B
        )

        x = self.conv_input(input_sp_tensor)

        res_dict['p1'] = self.conv1(x)
        res_dict['p2'] = self.conv2(res_dict['p1'])
        res_dict['p4'] = self.conv3(res_dict['p2'] )
        res_dict['p8'] = self.conv4(res_dict['p4'] )

        res_dict['p8_out'] = self.conv_out(res_dict['p8'])
        res_dict['bev'] = self.to_dense(res_dict['p8_out'])

        multi_scale_bev_feat = {}
        if hasattr(self, 'bev_neck'):
            res = self.bev_neck(res_dict['bev'])
            if isinstance(res, tuple):
                res_dict['bev'] = res[0]
                multi_scale_bev_feat = res[1]
            else:
                res_dict['bev'] = res
        if hasattr(self, 'bev_compressor'):
            res_dict['bev'] = self.bev_compressor(res_dict['bev'])

        out_dict = {}
        if 'voxel_feat' in self.scatter_keys:
            out_dict['voxel_feat'] = self.format_output(
                {k: res_dict[k] for k in self.cache_keys}, B)
        if 'bev_feat' in self.scatter_keys:
            out_dict['bev_feat'] = res_dict['bev']
        if 'multi_scale_bev_feat' in self.scatter_keys:
            multi_scale_bev_feat[1] = res_dict['bev']
            out_dict['multi_scale_bev_feat'] = \
                [{f'p{k * 8}': v[i] for k, v in multi_scale_bev_feat.items()} for i in range(B)]
        return out_dict

    def format_output(self, out_dict, B):
        out_list = []
        for i in range(B):
            new_dict = {}
            for k, v in out_dict.items():
                if isinstance(v, list) or isinstance(v, torch.Tensor):
                    new_dict[k] = v[i]
                else:
                    coor = v.indices
                    feat = v.features.contiguous()
                    mask = coor[:, 0] == i
                    new_dict[k] = {'coor': coor[mask, 1:], 'feat': feat[mask]}
            out_list.append(new_dict)

        return out_list

    def to_dense(self, stensor):
        spatial_features = stensor.dense()
        N, C, D, H, W = spatial_features.shape
        bev_featrues = spatial_features.reshape(N, C * D, H, W)
        return bev_featrues.contiguous()



