import torch
from torch import nn
from torch_scatter import scatter_add

from cosense3d.modules.utils import me_utils, common, misc, positional_encoding


class ObjFlow(nn.Module):
    def __init__(self, in_dim, data_info, stride=2, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.lidar_range = nn.Parameter(torch.Tensor(data_info['lidar_range']), requires_grad=False)
        self.voxel_size = data_info['voxel_size'][0]
        self.stride = stride
        self.convs = nn.Sequential(
            me_utils.minkconv_conv_block(in_dim, 128, 3, 1, 2, expand_coordinates=True),
            me_utils.minkconv_conv_block(128, 128, 3, 1, 2),
            me_utils.minkconv_conv_block(128, 128, 3, 1, 2),
            # me_utils.minkconv_conv_block(128, 128, 3, 1, 2),
            me_utils.minkconv_conv_block(128, in_dim, 3, 1, 2),
        )
        self.featurized_pe = misc.SELayer_Linear(in_dim)

    def forward(self, coor, feat):
        ref_ctr = (coor[..., :2] - self.voxel_size / 2) / (self.voxel_size * self.stride)
        ref_ctr = common.cat_coor_with_idx(ref_ctr)

        ref_pts_uniq, inverse_map = torch.unique(ref_ctr[..., :3].int(), dim=0, return_inverse=True)
        diff = (ref_pts_uniq[inverse_map, 1:3].float() - ref_ctr[..., 1:3])
        diff = (diff + 1) / 2
        diff_emb = positional_encoding.pos2posemb2d(diff, self.in_dim // 2)
        merged_feat = self.featurized_pe(diff_emb, feat.view(-1, self.in_dim))
        # print(merged_feat.shape, inverse_map.min(), inverse_map.max(), ref_pts_uniq.shape)
        merged_feat = scatter_add(merged_feat, index=inverse_map, dim=0)

        stensor = me_utils.ME.SparseTensor(
            coordinates=ref_pts_uniq.contiguous(),
            features=merged_feat,
        )
        stensor = self.convs(stensor)

        C = stensor.C.float()
        C[:, 1:3] = (self.voxel_size * self.stride) * C[:, 1:3] + self.voxel_size / 2
        C = torch.cat([C, torch.zeros_like(C[:, :1])], dim=1)
        return C, stensor.F
