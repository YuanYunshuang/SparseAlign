import functools
import torch

from cosense3d.modules import BaseModule, nn
from cosense3d.modules.utils.me_utils import mink_coor_limit, minkconv_conv_block, ME


class Spconv(nn.Module):
    def __init__(self, data_info, convs, d=2, dilation=False, **kwargs):
        super(Spconv, self).__init__()
        assert d == 2, 'only support dim=2'
        self.det_r = data_info.get('det_r', False)
        self.lidar_range = data_info.get('lidar_range', False)
        self.voxel_size = data_info['voxel_size']
        self.d = d
        self.dilation = dilation
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

    def forward(self, stensor_dict, **kwargs):
        out_dict = {}
        for k in self.convs:
            stride = int(k[1])
            stensor2d = self.get_2d_stensor(stensor_dict, stride)

            stensor2d = getattr(self, f'convs_{k}')(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed here.
            xylim = getattr(self, f'mink_xylim_{k}')
            mask = (stensor2d.C[:, 1] > xylim[0]) & (stensor2d.C[:, 1] <= xylim[1]) & \
                   (stensor2d.C[:, 2] > xylim[2]) & (stensor2d.C[:, 2] <= xylim[3])

            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]

            out_dict[k] = {
                'coor': coor,
                'feat': feat
            }
        return out_dict

    def get_2d_stensor(self, stensor_dict, stride):
        stensor = stensor_dict[f'p{stride}']
        if isinstance(stensor, ME.SparseTensor) and stensor.C.shape[-1] == 3:
            return stensor
        else:
            if isinstance(stensor, dict):
                coor, feat = stensor['coor'][:, :3], stensor['feat']
            elif isinstance(stensor, ME.SparseTensor):
                coor, feat = stensor.C[:, :3], stensor.F
            return ME.SparseTensor(
                coordinates=coor[:, :3].contiguous(),
                features=feat,
                tensor_stride=[stride] * 2
            )

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=self.d, bn_momentum=0.1,
        )
        in_dim = args['in_dim']
        out_dim = args['out_dim']
        layers = [minkconv_layer(in_dim, out_dim, args['kernels'][0], 1,
                                 expand_coordinates=self.dilation)]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(out_dim, out_dim, ks, 1,
                                         expand_coordinates=self.dilation))
        return nn.Sequential(*layers)