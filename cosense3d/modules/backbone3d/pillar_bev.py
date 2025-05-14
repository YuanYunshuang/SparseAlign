import torch
from torch import nn
from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import *
from cosense3d.modules.utils.me_utils import *


class PillarBEV(BaseModule):
    def __init__(self,
                 in_channels,
                 layer_nums,
                 layer_strides,
                 downsample_channels,
                 upsample_channels,
                 upsample_strides,
                 voxel_generator,
                 pillar_encoder,
                 bev_shrinker=None,
                 bev_compressor=None,
                 **kwargs):
        super(PillarBEV, self).__init__(**kwargs)
        self.pillar_encoder = plugin.build_plugin_module(pillar_encoder)
        self.voxel_generator = plugin.build_plugin_module(voxel_generator)
        self.grid_size = self.voxel_generator.grid_size

        if bev_shrinker is not None:
            self.bev_shrinker = plugin.build_plugin_module(bev_shrinker)
        if bev_compressor is not None:
            self.bev_compressor = plugin.build_plugin_module(bev_compressor)

        num_levels = len(layer_nums)
        c_in_list = [in_channels, *downsample_channels[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], downsample_channels[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(downsample_channels[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(downsample_channels[idx], downsample_channels[idx],
                              kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(downsample_channels[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            downsample_channels[idx], upsample_channels[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(upsample_channels[idx],
                                       eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            downsample_channels[idx], upsample_channels[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(upsample_channels[idx], eps=1e-3,
                                       momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(upsample_channels)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1],
                                   stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, points: list, **kwargs):
        N = len(points)
        voxels, coords, num_points = self.voxel_generator([x[:, :4] for x in points])
        coords = self.cat_data_from_list(coords, pad_idx=True)
        voxels = self.cat_data_from_list(voxels)
        num_points = self.cat_data_from_list(num_points)
        pillar_features = self.pillar_encoder(voxels, coords, num_points)
        bev_feat = self.to_dense_bev(coords, pillar_features, N)

        ups = []
        ret_dict = {}
        x = bev_feat

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(bev_feat.shape[2] / x.shape[2])
            ret_dict[f'p{stride}'] = x

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        if hasattr(self, 'bev_shrinker'):
            x = self.bev_shrinker(x)
        if hasattr(self, 'bev_compressor'):
            x = self.bev_compressor(x)

        out = {self.scatter_keys[0]: x}
        if 'multi_scale_bev_feat' in self.scatter_keys:
            stride = int(bev_feat.shape[2] / x.shape[2])
            ret_dict[f'p{stride}'] = x
            out['multi_scale_bev_feat'] = [{k: v[i] for k, v in ret_dict.items()} for i in range(N)]
        return out

    def format_output(self, res, N):
        out_dict = {self.scatter_keys[0]: self.decompose_stensor(res, N)}
        return out_dict

    def to_dense_bev(self, coor, feat, N):
        bev_feat = torch.zeros(N,
                               self.grid_size[2],
                               self.grid_size[1],
                               self.grid_size[0],
                               feat.shape[-1],
                               dtype=feat.dtype,
                               device=feat.device)
        coor = coor.long()
        bev_feat[coor[:, 0], coor[:, 1], coor[:, 2], coor[:, 3]] = feat
        bev_feat = bev_feat.permute(0, 4, 1, 2, 3)
        assert bev_feat.shape[2] == 1
        return bev_feat.squeeze(dim=2)



