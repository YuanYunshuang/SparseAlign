import torch
from torch import nn
import torch.nn.functional as F

from cosense3d.modules.utils.conv import ConvModule
from cosense3d.modules.utils.init import xavier_init


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(
                inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2,
                                                  1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarEncoder(nn.Module):
    def __init__(self,
                 features,
                 voxel_size,
                 lidar_range,
                 channels,
                 use_norm=True):
        super(PillarEncoder, self).__init__()
        self.voxel_size = nn.Parameter(torch.tensor(voxel_size), requires_grad=False)
        self.lidar_range = nn.Parameter(torch.tensor(lidar_range), requires_grad=False)
        self.offset = nn.Parameter(self.voxel_size / 2 + self.lidar_range[:3],
                                   requires_grad=False)
        self.num_point_features = sum(
            [getattr(self, f"{f}_dim") for f in features])
        self.features = features
        assert isinstance(channels, list)
        self.channels = [self.num_point_features] + channels
        self.out_channels = channels[-1]
        self.use_norm = use_norm
        self._init_layers(self.channels)

    def _init_layers(self, channels):
        pfn_layers = []
        for i in range(len(channels) - 1):
            in_filters = channels[i]
            out_filters = channels[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(channels) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, voxel_features, coords, voxel_num_points):
        points_mean = voxel_features[..., :3].sum(dim=1, keepdim=True) / \
            voxel_num_points.view(-1, 1, 1)
        f_cluster = voxel_features[..., :3] - points_mean

        coords_metric = coords[:, [3, 2, 1]].unsqueeze(1) * self.voxel_size + self.offset
        f_center = voxel_features[..., :3] - coords_metric

        features = self.compose_voxel_feature(voxel_features) + [f_cluster, f_center]
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        features *= mask.unsqueeze(-1)
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        return features

    def compose_voxel_feature(self, voxel_features):
        features = []
        if 'absolute_xyz' in self.features:
            features.append(voxel_features[..., :3])
        if 'distance' in self.features:
            features.append(torch.norm(voxel_features[..., :3], 2, -1,
                                       keepdim=True))
        if 'intensity' in self.features:
            assert voxel_features.shape[-1] >= 4
            features.append(voxel_features[..., 3:4])
        return features

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num,
                               dtype=torch.int,
                               device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    @property
    def distance_dim(self):
        return 1

    @property
    def absolute_xyz_dim(self):
        return 6

    @property
    def xyz_dim(self):
        return 3
    @property
    def intensity_dim(self):
        return 1
