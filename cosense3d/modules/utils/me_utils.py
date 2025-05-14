import torch
from torch import nn
import MinkowskiEngine as ME
from MinkowskiEngine.MinkowskiKernelGenerator import KernelGenerator


@torch.no_grad()
def metric2indices(coor, voxel_size):
    """"Round towards floor"""
    indices = coor.clone()
    if isinstance(voxel_size, float):
        indices[:, 1:3] = indices[:, 1:3] / voxel_size
    else:
        indices[:, 1] = indices[:, 1] / voxel_size[0]
        indices[:, 2] = indices[:, 2] / voxel_size[1]
    return torch.floor(indices).long()


@torch.no_grad()
def indices2metric(indices, voxel_size):
    """Voxel indices to voxel center in meter"""
    coor = indices.clone().float()
    coor[:, 1] = (coor[:, 1] + 0.5) * voxel_size[0]
    coor[:, 2] = (coor[:, 2] + 0.5) * voxel_size[1]
    return coor


@torch.no_grad()
def mink_coor_limit(lidar_range, voxel_size, stride):
    if not isinstance(voxel_size, list):
        voxel_size = [voxel_size, voxel_size]
    lr = lidar_range
    x_max = (round(lr[3] / voxel_size[0]) - 1) // stride * stride  # relevant to ME
    x_min = (round(lr[0] / voxel_size[0]) + 1) // stride * stride - stride  # relevant to ME
    y_max = (round(lr[4] / voxel_size[1]) - 1) // stride * stride
    y_min = (round(lr[1] / voxel_size[1]) + 1) // stride * stride - stride
    return [x_min, x_max, y_min, y_max]


def update_me_essentials(self: object, data_info: dict, stride: int=None):
    """Update essential variables for ME-based models

    :param self: instance of a python class
    :param data_info:
        - det_r: float
        - lidar_range: [xmin, ymin, zmin, xmax, ymax, zmax]
        - voxel_size: [vx, vy, vz]
    :param stride:
    :return:
    """
    for k, v in data_info.items():
        setattr(self, k, v)

    if getattr(self, 'det_r', False):
        lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
    elif getattr(self, 'lidar_range', False):
        lr = self.lidar_range
    else:
        raise NotImplementedError
    setattr(self, 'lidar_range', lr)

    if stride is not None:
        setattr(self, 'stride', stride)
        setattr(self, 'res', (self.stride * self.voxel_size[0], self.stride * self.voxel_size[1]))
        setattr(self, 'mink_xylim', mink_coor_limit(lr, self.voxel_size, self.stride))
        setattr(self, 'size_x', round((lr[3] - lr[0]) / self.res[0]))
        setattr(self, 'size_y', round((lr[4] - lr[1]) / self.res[1]))
        setattr(self, 'offset_sz_x', round(lr[0] / self.res[0]))
        setattr(self, 'offset_sz_y', round(lr[1] / self.res[1]))


@torch.no_grad()
def me_coor_to_grid_indices(lr, voxel_size, stride, coor):
    res_x, res_y = stride * voxel_size[0], stride * voxel_size[1]
    size_x = round((lr[3] - lr[0]) / res_x)
    size_y = round((lr[4] - lr[1]) / res_y)
    offset_sz_x = round(lr[0] / res_x)
    offset_sz_y = round(lr[1] / res_y)
    inds = coor.clone()
    inds[:, 0] -= offset_sz_x
    inds[:, 1] -= offset_sz_y
    in_range_mask = (inds >= 0).all(dim=-1) & inds[:, 0] < size_x & inds[:, 1] < size_y
    return inds, in_range_mask


@torch.no_grad()
def bev_sparse_to_dense(self, preds):
    conf, unc = preds['conf'], preds['unc'],
    ctrs = preds['centers'][:, :3]  # N 2
    batch_size = ctrs[:, 0].max().int() + 1
    conf_map = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                           device=conf.device)
    unc_map = torch.ones((batch_size, self.size_x, self.size_y),
                         device=unc.device)
    inds = metric2indices(ctrs, self.res).T
    inds[1] -= self.offset_sz_x
    inds[2] -= self.offset_sz_y
    conf_map[inds[0], inds[1], inds[2]] = conf
    unc_map[inds[0], inds[1], inds[2]] = unc
    return conf_map, unc_map


def minkconv_layer(in_dim, out_dim, kernel, stride, d, tr=False):
    if not isinstance(kernel, list):
        kernel = [kernel] * d
    else:
        assert len(kernel) == d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=1,
        dimension=d
    )
    return conv_layer


def minkconv_conv_block(in_dim, out_dim, kernel, stride,
                        d=3,
                        bn_momentum=0.1,
                        activation='LeakyReLU',
                        tr=False,
                        expand_coordinates=False,
                        norm_before=False,
                        dilation=1,
                        distributed=False):
    if isinstance(kernel, int):
        kernel = [kernel] * d
    if isinstance(stride, int):
        stride = [stride] * d
    if tr:
        conv = getattr(ME, 'MinkowskiConvolutionTranspose')
    else:
        conv = getattr(ME, 'MinkowskiConvolution')
    conv_layer = conv(
        in_channels=in_dim,
        out_channels=out_dim,
        kernel_size=kernel,
        stride=stride,
        dilation=dilation,
        dimension=d,
        expand_coordinates=expand_coordinates
    )
    activation_fn = getattr(ME, f'Minkowski{activation}')()
    if distributed:
        norm_layer = ME.MinkowskiSyncBatchNorm(out_dim, momentum=bn_momentum)
    else:
        norm_layer = ME.MinkowskiBatchNorm(out_dim, momentum=bn_momentum)
    if norm_before:
        layer = nn.Sequential(conv_layer, norm_layer, activation_fn)
    else:
        layer = nn.Sequential(conv_layer, activation_fn, norm_layer)
    return layer


def get_conv_block(nc, k=3, d=3, s=[2, 1, 1], tr=False, bn_momentum=0.1,
                   expand=[False,False,False], dilation=[1, 1, 1], distributed=False):
    """
    create sparse convolution block
    :param nc: number of channels in each layer in [in_layer, mid_layer, out_layer]
    :param k: kernel size
    :param s: strides
    :param tr: transposed convolution
    :return: conv block
    """
    if isinstance(k, int):
        k = [k,] * d
    else:
        assert len(k) == d
    bnm = bn_momentum
    assert len(nc) == 3
    return nn.Sequential(
            minkconv_conv_block(nc[0], nc[1], k, s[0], d, bnm, tr=tr,
                                distributed=distributed,
                                expand_coordinates=expand[0],
                                dilation=dilation[0]),
            minkconv_conv_block(nc[1], nc[1], k, s[1], d, bnm, tr=tr,
                                distributed=distributed,
                                expand_coordinates=expand[1],
                                dilation=dilation[1]),
            minkconv_conv_block(nc[1], nc[2], k, s[2], d, bnm, tr=tr,
                                distributed=distributed,
                                expand_coordinates=expand[2],
                                dilation=dilation[2]),
        )


def sparse_to_dense(stensor, voxel_size, det_r):
    b = int(stensor.C[:, 0].max()) + 1
    d = stensor.F.shape[-1]
    stride = stensor.tensor_stride
    h = int((det_r['x'][1] - det_r['x'][0]) / voxel_size[0]) // stride[0]
    w = int((det_r['y'][1] - det_r['y'][0]) / voxel_size[1]) // stride[1]
    x_offset = int(det_r['x'][0] / voxel_size[0])
    y_offset = int(det_r['y'][0] / voxel_size[1])
    assert len(stensor.C[:, 3].unique()) == 1
    dtensor = stensor.dense(
        shape=torch.Size((b, d, h, w, 1)),
        min_coordinate=torch.Tensor([x_offset, y_offset, 0]).int())[0].squeeze(dim=-1)

    return dtensor


def prepare_input_data(points_list, voxel_size, QMODE, floor_height,
                       coor_dim=3, feat_dim=3):
    device = points_list[0].device
    coords = []
    features = []
    vs = torch.tensor(voxel_size).reshape(1, 3).to(device)
    for i, points in enumerate(points_list):
        pts = points.clone()
        if floor_height == "adaptive":
            pts[:, 3] -= (pts[:, 3].min() - 1e-4)
        elif isinstance(floor_height, float):
            pts[:, 3] -= floor_height
        pts[:, 1:4] = pts[:, 1:4] / vs
        features.append(points[:, 1:feat_dim + 1])
        coords.append(pts)
    coords = torch.cat(coords, dim=0)
    features = torch.cat(features, dim=0)

    x = ME.TensorField(
        features=features.contiguous(),
        coordinates=coords[:, :coor_dim + 1].contiguous(),
        quantization_mode=QMODE,
        device=device
    )
    # ME rounds to the floor when casting coords to integer
    return x


def voxelize_with_centroids(x: ME.TensorField, enc_mlp, pc_range):
    cm = x.coordinate_manager
    features = x.F
    coords = x.C[:, 1:]

    out = x.sparse()
    size = torch.Size([len(out), len(x)])
    tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
    coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
    features_p1, _ = downsample_points(features, tensor_map, field_map, size)
    if len(features) != len(tensor_map):
        print('ME: features != tensor map')
    norm_features = normalize_points(features, features_p1, tensor_map)

    features[:, :3] = (features[:, :3] - pc_range[:3]) / (pc_range[3:] - pc_range[:3])
    voxel_embs = enc_mlp(torch.cat([features, norm_features], dim=1))
    down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="avg")
    out = ME.SparseTensor(features=down_voxel_embs,
                          coordinate_map_key=out.coordinate_key,
                          coordinate_manager=cm)

    norm_points_p1 = normalize_centroids(coords_p1, out.C, out.tensor_stride[0])
    return out, norm_points_p1, features_p1, count_p1, voxel_embs


def devoxelize_with_centroids(out: ME.SparseTensor, x: ME.TensorField, h_embs):
    feats = torch.cat([out.slice(x).F, h_embs], dim=1)
    return feats


@torch.no_grad()
def normalize_points(points, centroids, tensor_map):
    tensor_map = tensor_map if tensor_map.dtype == torch.int64 else tensor_map.long()
    norm_points = points - centroids[tensor_map]
    return norm_points


@torch.no_grad()
def normalize_centroids(down_points, coordinates, tensor_stride):
    norm_points = (down_points - coordinates[:, 1:]) / tensor_stride - 0.5
    return norm_points


@torch.no_grad()
def get_kernel_map_and_out_key(stensor, stensor_out=None,
                               kernel_size=3, stride=1, dilation=1,
                               kernel_type='cube', kernel_generator=None):
    """
    Generate kernel maps for the input stensor.
    The hybrid and custom kernel is not implemented in ME v0.5.x,
    this function uses a kernel mask to select the kernel maps for
    the customized kernel shapes.
    :param stensor: ME.SparseTensor, NxC
    :param kernel_type: 'cube'(default) | 'hybrid'
    :return: masked kernel maps
    """
    D = stensor.C.shape[-1] - 1
    if kernel_generator is None:
        kernel_generator = KernelGenerator(kernel_size=kernel_size,
                                           stride=stride,
                                           dilation=dilation,
                                           dimension=D)
    assert D == len(kernel_generator.kernel_stride)
    cm = stensor.coordinate_manager
    in_key = stensor.coordinate_key
    if stensor_out is None:
        out_key = cm.stride(in_key, kernel_generator.kernel_stride)
    else:
        out_key = stensor_out.coordinate_key
    region_type, region_offset, _ = kernel_generator.get_kernel(
        stensor.tensor_stride, False)

    kernel_map = cm.kernel_map(in_key,
                               out_key,
                               kernel_generator.kernel_stride,
                               kernel_generator.kernel_size,
                               kernel_generator.kernel_dilation,
                               region_type=region_type,
                               region_offset=region_offset)
    if kernel_type=='cube':
        kernel_volume = kernel_generator.kernel_volume
    elif kernel_type=='hybrid':
        assert dilation == 1, "currently, hybrid kernel only support dilation=1."
        xx = torch.tensor([-1, 0, 1]).int()
        xx_list = [xx for i in range(D)]
        kernels = torch.meshgrid([*xx_list], indexing='ij')
        kernels = torch.stack([t.flatten() for t in kernels], dim=1)
        kernel_mask = torch.zeros_like(kernels[:, 0]).bool()
        m = torch.logical_or(
            kernels[:, 0] == 0,
            torch.logical_and(kernels[:, 0]==-1, (kernels[:, 1:]==0).all(dim=1))
        )
        kernel_mask[m] = True
        kernel_mask_map = {ic.item(): ih for ih, ic in enumerate(torch.where(kernel_mask)[0])}
        kernel_map = {kernel_mask_map[k]: v for k, v in kernel_map.items() if kernel_mask[k]}
        kernel_volume = kernel_mask.sum().item()
    else:
        raise NotImplementedError

    return kernel_map, out_key, kernel_volume


@torch.no_grad()
def downsample_points(points, tensor_map, field_map, size):
    down_points = ME.MinkowskiSPMMAverageFunction().apply(
        tensor_map, field_map, size, points
    )
    _, counts = torch.unique(tensor_map, return_counts=True)
    return down_points, counts.unsqueeze_(1).type_as(down_points)


@torch.no_grad()
def stride_centroids(points, counts, rows, cols, size):
    stride_centroids = ME.MinkowskiSPMMFunction().apply(rows, cols, counts, size, points)
    ones = torch.ones(size[1], dtype=points.dtype, device=points.device)
    stride_counts = ME.MinkowskiSPMMFunction().apply(rows, cols, ones, size, counts)
    stride_counts.clamp_(min=1)
    return torch.true_divide(stride_centroids, stride_counts), stride_counts


def downsample_embeddings(embeddings, inverse_map, size, mode="avg"):
    assert len(embeddings) == size[1]
    assert mode in ["avg", "max"]
    if mode == "max":
        in_map = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiDirectMaxPoolingFunction().apply(
            in_map, inverse_map, embeddings, size[0]
        )
    else:
        cols = torch.arange(size[1], dtype=inverse_map.dtype, device=inverse_map.device)
        down_embeddings = ME.MinkowskiSPMMAverageFunction().apply(
            inverse_map, cols, size, embeddings
        )
    return down_embeddings








