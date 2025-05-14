import torch
from torch import nn
from cosense3d.modules import BaseModule
from cosense3d.modules.utils.common import *
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.plugin.attn import VerticalAttention


class MinkUnet(BaseModule):
    QMODE = ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    def __init__(self,
                 data_info,
                 stride,
                 in_dim,
                 d=3,
                 kernel_size_layer1=3,
                 enc_dim=32,
                 cache_strides=None,
                 floor_height=None,
                 height_attn=None,
                 height_compression=None,
                 compression_kernel_size_xy=1,
                 to_dense=False,
                 dist=False,
                 **kwargs):
        super(MinkUnet, self).__init__(**kwargs)
        update_me_essentials(self, data_info)
        self.stride = stride
        self.in_dim = in_dim
        self.enc_dim = enc_dim
        self.floor_height = floor_height
        self.height_attn = height_attn
        self.to_dense = to_dense
        self.height_compression = height_compression
        self.compression_kernel_size_xy = compression_kernel_size_xy
        self.d = d
        self.lidar_range_tensor = nn.Parameter(torch.Tensor(self.lidar_range), requires_grad=False)
        # For determine batchnorm type: if the model is trained on multiple GPUs with ME.MinkowskiBatchNorm,
        # the BN would perform differently in eval mode because the running_mean and running_var would be
        # different to training mode, this is caused by different number of tracked batches, therefore if
        # ditributed training is used for this model, either ME.MinkowskiSyncBatchNorm should be used, or
        # the running mean and var should be adapted.
        # TODO: adapt running mean and var in inference mode if model is trained with DDP
        self.dist = dist
        if cache_strides is None:
            self.cache_strides = [stride]
            self.max_resolution = stride
        else:
            self.max_resolution = min(cache_strides)
            self.cache_strides = cache_strides
        self._init_unet_layers(kernel_size_layer1)
        if height_compression is not None:
            self._init_height_compression_layers(height_compression)
        if height_attn is not None:
            self._init_height_attn_layers(height_attn)
        self.init_weights()

    def _init_unet_layers(self, kernel_size_layer1=3):
        self.enc_mlp = linear_layers([self.in_dim * 2, 16, self.enc_dim], norm='LN')
        kernel_conv1 = [kernel_size_layer1,] * min(self.d, 3)
        kernel = [3,] * min(self.d, 3)
        if self.d == 4:
            kernel = kernel + [1,]
            kernel_conv1 = kernel + [1,]

        kwargs = {'d': self.d, 'bn_momentum': 0.1}
        self.conv1 = minkconv_conv_block(self.enc_dim, self.enc_dim, kernel_conv1,
                                         1, **kwargs)
        self.conv2 = get_conv_block([self.enc_dim, self.enc_dim, self.enc_dim], kernel, **kwargs)
        self.conv3 = get_conv_block([self.enc_dim, self.enc_dim * 2, self.enc_dim * 2], kernel, **kwargs)
        self.conv4 = get_conv_block([self.enc_dim * 2, self.enc_dim * 4, self.enc_dim * 4], kernel, **kwargs)

        if self.max_resolution <= 4:
            self.trconv4 = get_conv_block([self.enc_dim * 4, self.enc_dim * 2, self.enc_dim * 2], kernel, tr=True, **kwargs)
        if self.max_resolution <= 2:
            self.trconv3 = get_conv_block([self.enc_dim * 4, self.enc_dim * 2, self.enc_dim * 2], kernel, tr=True, **kwargs)
        if self.max_resolution <= 1:
            self.trconv2 = get_conv_block([self.enc_dim * 3, self.enc_dim * 2, self.enc_dim], kernel, tr=True, **kwargs)
            self.out_layer = minkconv_layer(self.enc_dim * 2, self.enc_dim, kernel, 1, d=self.d)

    def _init_height_compression_layers(self, planes):
        self.stride_size_dict = {}
        for k, v in planes.items():
            self.stride_size_dict[int(k[1])] = self.grid_size(int(k[1]))
            layers = []
            steps = v['steps']
            channels = v['channels']
            for i, s in enumerate(steps):
                kernel = [self.compression_kernel_size_xy] * 2 + [s]
                stride = [1] * 2 + [s]
                layers.append(
                    minkconv_conv_block(channels[i], channels[i+1],
                                        kernel, stride, self.d, 0.1)
                )
            layers = nn.Sequential(*layers)
            setattr(self, f'{k}_compression', layers)

    def _init_height_attn_layers(self, planes):
        self.attn_list = []
        for k, v in planes.items():
            self.attn_list.append(k)
            setattr(self, f'{k}_attn', VerticalAttention(v[0], v[1]))

    def init_weights(self):
        for n, p in self.named_parameters():
            if ('mlp' in n and 'weight' in n) or 'kernel' in n:
                if p.ndim == 1:
                    continue
                nn.init.xavier_uniform_(p)

    def to_gpu(self, gpu_id):
        self.to(gpu_id)
        return ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm

    def forward(self, points: list, **kwargs):
        res = self.forward_unet(points, **kwargs)

        if self.height_compression is not None:
            res = self.forward_height_compression(res)
        elif self.height_attn is not None:
            res = self.forward_height_attn(res)

        res = self.format_output(res, len(points))
        return res

    def forward_unet(self, points, **kwargs):
        N = len(points)
        points = [torch.cat([torch.ones_like(pts[:, :1]) * i, pts], dim=-1
                            ) for i, pts in enumerate(points)]
        x = prepare_input_data(points, self.voxel_size, self.QMODE, self.floor_height,
                               self.d, self.in_dim)
        x1, norm_points_p1, points_p1, count_p1, pos_embs = voxelize_with_centroids(
            x, self.enc_mlp, self.lidar_range_tensor)

        # convs
        x1 = self.conv1(x1)
        x2 = self.conv2(x1)
        x4 = self.conv3(x2)
        p8 = self.conv4(x4)
        p8_cat = p8

        # transposed convs
        if self.max_resolution <= 4:
            p4 = self.trconv4(p8)
            p4_cat = ME.cat(x4, p4)
        if self.max_resolution <= 2:
            p2 = self.trconv3(p4_cat)
            p2_cat = ME.cat(x2, p2)
        if self.max_resolution <= 1:
            p1 = self.trconv2(p2_cat)
            p1_cat = self.out_layer(ME.cat(x1, p1))
        if self.max_resolution == 0:
            p0 = devoxelize_with_centroids(p1, x, pos_embs)
            p0_cat = {'coor': torch.cat(points, dim=0), 'feat': p0}

        vars = locals()
        res = {f'p{k}': vars[f'p{k}_cat'] for k in self.cache_strides}

        tmp = x4.F.max(dim=0).values
        return res

    def forward_height_compression(self, res):
        for stride in self.stride_size_dict.keys():
            out_tensor = getattr(self, f'p{stride}_compression')(res[f'p{stride}'])
            assert len(out_tensor.C[:, 3].unique()) == 1, \
                (f"height is not fully compressed. "
                 f"Unique z coords: {','.join([str(x.item()) for x in out_tensor.C[:, 3].unique()])}")
            if self.to_dense:
                out_tensor = self.stensor_to_dense(out_tensor).permute(0, 3, 1, 2)
                res[f'p{stride}'] = out_tensor
            else:
                ctr = indices2metric(out_tensor.C, self.voxel_size)
                res[f'p{stride}'] = {'coor': out_tensor.C[:, :3], 'feat': out_tensor.F, 'ctr': ctr[:, 1:3]}
        return res

    def forward_height_attn(self, res):
        for k in self.attn_list:
            C, F = getattr(self, f'{k}_attn')(res[k])
            if self.to_dense:
                raise NotImplementedError
            else:
                ctr = indices2metric(C, self.voxel_size)
                res[k] = {'coor': C, 'feat': F, 'ctr': ctr[:, 1:3]}
        return res

    def format_output(self, res, N):
        out_dict = {self.scatter_keys[0]: self.decompose_stensor(res, N)}
        return out_dict

    def stensor_to_dense(self, stensor):
        mask, indices = self.valid_coords(stensor)
        b = int(stensor.C[:, 0].max()) + 1
        d = stensor.F.shape[1]
        features = stensor.F[mask].view(-1, d)
        s = self.stride_size_dict[stensor.tensor_stride[0]]
        dtensor = features.new_zeros((b, s[0], s[1], d))
        dtensor[indices[0], indices[1], indices[2]] = features
        return dtensor

    def valid_coords(self, stensor):
        stride = stensor.tensor_stride
        s = self.stride_size_dict[stride[0]]
        # remove voxels that are outside range
        xi = torch.div(stensor.C[:, 1], stride[0], rounding_mode='floor') + s[0] / 2
        yi = torch.div(stensor.C[:, 2], stride[1], rounding_mode='floor') + s[1] / 2

        mask = (xi >= 0) * (xi < s[0]) * (yi >= 0) * (yi < s[1])
        indices = (stensor.C[:, 0][mask].long(),
                   xi[mask].long(),
                   yi[mask].long()
                   )
        # if the backbone uses 4d convs, last dim is time
        if stensor.C.shape[1] == 5:
            ti = stensor.C[:, 4]
            mask = mask * (ti >= 0) * (ti < self.seq_len)
            indices = indices + ti[mask].long()
        return mask, indices

    def grid_size(self, stride):
        x_range = self.lidar_range[3] - self.lidar_range[0]
        y_range = self.lidar_range[4] - self.lidar_range[1]
        x_size = int(x_range / self.voxel_size[0]) // stride
        y_size = int(y_range / self.voxel_size[1]) // stride
        return (x_size, y_size)





