import torch
from torch import nn
from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import *
from cosense3d.modules.utils.me_utils import *


class VoxelNet(BaseModule):
    def __init__(self,
                 voxel_generator,
                 voxel_encoder,
                 cml,
                 neck=None,
                 bev_compressor=None,
                 **kwargs):
        super(VoxelNet, self).__init__(**kwargs)
        self.voxel_generator = plugin.build_plugin_module(voxel_generator)
        self.voxel_encoder = plugin.build_plugin_module(voxel_encoder)
        self.grid_size = self.voxel_generator.grid_size
        self.cml = plugin.build_plugin_module(cml)

        if neck is not None:
            self.neck = plugin.build_plugin_module(neck)
        if bev_compressor is not None:
            self.bev_compressor = plugin.build_plugin_module(bev_compressor)

    def forward(self, points: list, **kwargs):
        N = len(points)
        voxels, coords, num_points = self.voxel_generator(points)
        coords = self.cat_data_from_list(coords, pad_idx=True)
        voxels = self.cat_data_from_list(voxels)
        num_points = self.cat_data_from_list(num_points)
        voxel_features = self.voxel_encoder(voxels, coords, num_points)
        if self.cml.dense:
            voxel_features = self.to_dense(coords, voxel_features, N)
            voxel_features = self.cml(voxel_features)
        else:
            voxel_features, voxel_coords = self.cml(voxel_features, coords)
            voxel_features = self.to_dense(voxel_coords, voxel_features, N, filter_range=True)
        # 3d to 2d feature
        bev_feat = voxel_features.flatten(1, 2)
        x = bev_feat
        ret_dict = {}
        if hasattr(self, 'neck'):
            res = self.neck(x)
            if isinstance(res, torch.Tensor):
                x = res
            else:
                x = res[0]
                ret_dict = res[1]
        if hasattr(self, 'bev_compressor'):
            x = self.bev_compressor(x)

        out = {self.scatter_keys[0]: x}
        if 'multi_scale_bev_feat' in self.scatter_keys:
            stride = int(bev_feat.shape[2] / x.shape[2])
            ret_dict[f'p{stride}'] = x
            out['multi_scale_bev_feat'] = [{k: v[i] for k, v in ret_dict.items()} for i in range(N)]

        return out

    def to_dense(self, coor, feat, N, filter_range=False):
        if filter_range:
            strides = self.cml.out_strides.cpu()
            grid_size = torch.ceil(self.grid_size[[2, 1, 0]] / strides).int().tolist()
            mask = (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[0]) & \
                   (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[1]) & \
                   (coor[:, 3] >= 0) & (coor[:, 3] < grid_size[2])
            coor, feat = coor[mask], feat[mask]
        else:
            grid_size = self.grid_size[[2, 1, 0]].tolist()
        bev_feat = torch.zeros(N,
                               grid_size[0],
                               grid_size[1],
                               grid_size[2],
                               feat.shape[-1],
                               dtype=feat.dtype,
                               device=feat.device)
        coor = coor.long()
        bev_feat[coor[:, 0], coor[:, 1], coor[:, 2], coor[:, 3]] = feat

        return bev_feat.permute(0, 4, 1, 2, 3)



