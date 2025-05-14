import math
from collections import OrderedDict


def get_minkunet_cfg(gather_keys, scatter_keys, voxel_size, point_cloud_range,
                     in_dim=4, dim=3, out_stride=2, height_compression=[2, 8],
                     compression_kernel_size_xy=1, cache_strides=[2, 8], enc_dim=32,
                     kernel_size_layer1=5, floor_height=None, freeze=False):
    data_info = dict(lidar_range=point_cloud_range, voxel_size=voxel_size)
    if len(height_compression) > 0:
        hc = OrderedDict()
        height = (point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]
        dims = {1: enc_dim, 2: enc_dim * 3, 4: enc_dim * 4, 8: enc_dim * 4}
        # dims = {1: enc_dim, 2: enc_dim * 4, 4: enc_dim * 4, 8: enc_dim * 4}
        for stride in height_compression:
            downx = math.ceil(height / stride)
            dim = dims[stride]
            if downx > 4:
                hc[f'p{stride}'] = dict(channels=[dim, 256, 384], steps=[5, max(downx // 5, 2)])
            else:
                hc[f'p{stride}'] = dict(channels=[dim, 256], steps=[downx])
    else:
        hc = None
    floor_height = point_cloud_range[2] if floor_height is None else floor_height
    return dict(
        type='backbone3d.mink_unet.MinkUnet',
        freeze=freeze,
        gather_keys=gather_keys,
        scatter_keys=scatter_keys,
        d=3,
        cache_strides=cache_strides,
        kernel_size_layer1=kernel_size_layer1,
        in_dim=in_dim,
        stride=out_stride,
        floor_height=floor_height,
        data_info=data_info,
        height_compression=hc,
        compression_kernel_size_xy=compression_kernel_size_xy,
        enc_dim=enc_dim,
    )


