import functools
import torch

from cosense3d.modules import BaseModule, nn
from cosense3d.modules.utils.me_utils import mink_coor_limit, minkconv_conv_block, ME, indices2metric


class DilationSpconv(BaseModule):
    def __init__(self, data_info, convs, d=2, n_layers=None, mask_shape='rectangle', **kwargs):
        super(DilationSpconv, self).__init__(**kwargs)
        self.det_r = data_info.get('det_r', False)
        self.lidar_range = data_info.get('lidar_range', False)
        self.voxel_size = data_info['voxel_size']
        self.d = d
        self.n_layers = n_layers
        self.mask_shape = mask_shape
        self.conv_args = convs
        self.convs = []
        for k, conv_args in convs.items():
            self.convs.append(k)
            setattr(self, f'convs_{k}', self.get_conv_layer(conv_args))
            stride = int(k[1])

            if self.det_r:
                lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
            elif self.lidar_range:
                lr = self.lidar_range
            else:
                raise NotImplementedError
            setattr(self, f'mink_xylim_{k}', mink_coor_limit(lr, self.voxel_size, stride))  # relevant to ME

    def to_gpu(self, gpu_id):
        self.to(gpu_id)
        return ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm

    def forward(self, stensor_list, **kwargs):
        out_dict = {}
        for k in self.convs:
            stride = int(k[1])
            coor, feat, ctr = self.compose_stensor(stensor_list, stride)
            stensor2d = ME.SparseTensor(
                coordinates=coor[:, :3].contiguous(),
                features=feat,
                tensor_stride=[stride] * 2
            )

            stensor2d = getattr(self, f'convs_{k}')(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed.
            xylim = getattr(self, f'mink_xylim_{k}')
            if self.mask_shape == 'rectangle':
                mask = (stensor2d.C[:, 1] > xylim[0]) & (stensor2d.C[:, 1] <= xylim[1]) & \
                       (stensor2d.C[:, 2] > xylim[2]) & (stensor2d.C[:, 2] <= xylim[3])
            elif self.mask_shape == 'circle':
                mask = (stensor2d.C[:, 1:3].abs() <= xylim[1]).all(dim=1)
            else:
                mask = torch.ones_like(stensor2d.C[:, 0]).bool()

            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]
            ctr = indices2metric(coor, self.voxel_size)[:, 1:]

            out_dict[k] = {
                'coor': coor,
                'feat': feat,
                'ctr': ctr
            }
        return self.format_output(out_dict, len(stensor_list))

    def format_output(self, out_dict, B):
        out_list = self.decompose_stensor(out_dict, B)
        return {self.scatter_keys[0]: out_list}

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=self.d, bn_momentum=0.1,
        )
        in_dim = args['in_dim']
        out_dim = args['out_dim']
        layers = [minkconv_layer(in_dim, out_dim, args['kernels'][0], 1,
                                 expand_coordinates=True)]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(out_dim, out_dim, ks, 1,
                                         expand_coordinates=True))
        if self.n_layers is not None and self.n_layers > len(args['kernels']):
            for _ in range(self.n_layers - len(args['kernels'])):
                layers.append(minkconv_layer(out_dim, out_dim, 3, 1,
                                             expand_coordinates=False))
        return nn.Sequential(*layers)


class DilationSpconvAblation(BaseModule):
    def __init__(self, data_info, convs, d=2, n_layers=None, **kwargs):
        super(DilationSpconvAblation, self).__init__(**kwargs)
        self.det_r = data_info.get('det_r', False)
        self.lidar_range = data_info.get('lidar_range', False)
        self.voxel_size = data_info['voxel_size']
        self.d = d
        self.n_layers = n_layers
        self.conv_args = convs
        self.convs = []
        for k, conv_args in convs.items():
            self.convs.append(k)
            setattr(self, f'convs_{k}', self.get_conv_layer(conv_args))
            stride = int(k[1])

            if self.det_r:
                lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
            elif self.lidar_range:
                lr = self.lidar_range
            else:
                raise NotImplementedError
            setattr(self, f'mink_xylim_{k}', mink_coor_limit(lr, self.voxel_size, stride))  # relevant to ME

    def to_gpu(self, gpu_id):
        self.to(gpu_id)
        return ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm

    def forward(self, stensor_list, **kwargs):
        out_dict = {}
        for k in self.convs:
            stride = int(k[1])
            coor, feat, ctr = self.compose_stensor(stensor_list, stride)
            stensor2d = ME.SparseTensor(
                coordinates=coor[:, :3].contiguous(),
                features=feat,
                tensor_stride=[stride] * 2
            )

            stensor2d = getattr(self, f'convs_{k}')(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed here.
            xylim = getattr(self, f'mink_xylim_{k}')
            mask = (stensor2d.C[:, 1] > xylim[0]) & (stensor2d.C[:, 1] <= xylim[1]) & \
                   (stensor2d.C[:, 2] > xylim[2]) & (stensor2d.C[:, 2] <= xylim[3])

            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]
            ctr = indices2metric(coor, self.voxel_size)[:, 1:]

            out_dict[k] = {
                'coor': coor,
                'feat': feat,
                'ctr': ctr
            }
        return self.format_output(out_dict, len(stensor_list))

    def format_output(self, out_dict, B):
        out_list = self.decompose_stensor(out_dict, B)
        return {self.scatter_keys[0]: out_list}

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=self.d, bn_momentum=0.1,
        )
        in_dim = args['in_dim']
        out_dim = args['out_dim']
        layers = [minkconv_layer(in_dim, out_dim, args['kernels'][0], 1,
                                 expand_coordinates=False)]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(out_dim, out_dim, ks, 1,
                                         expand_coordinates=False))
        if self.n_layers is not None and self.n_layers > len(args['kernels']):
            for _ in range(self.n_layers - len(args['kernels'])):
                layers.append(minkconv_layer(out_dim, out_dim, 3, 1,
                                             expand_coordinates=False))
        return nn.Sequential(*layers)