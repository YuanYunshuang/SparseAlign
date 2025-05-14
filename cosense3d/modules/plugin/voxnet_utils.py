# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet, modified by Yunshuang Yuan
# License: TDG-Attribution-NonCommercial-NoDistrib
# Modified by Yunshuang Yuan

import torch
from torch import nn
import torch.nn.functional as F
import MinkowskiEngine as ME


class Conv3d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)


class CML(nn.Module):
    def __init__(self, in_channels):
        super(CML, self).__init__()
        self.dense = True
        self.conv3d_1 = Conv3d(in_channels, in_channels, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.conv3d_2 = Conv3d(in_channels, in_channels, 3, s=(1, 1, 1), p=(0, 1, 1))
        self.conv3d_3 = Conv3d(in_channels, in_channels, 3, s=(2, 1, 1), p=(1, 1, 1))
        self.out_strides = (4, 1, 1)

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)
        return x


class CMLSparse(nn.Module):
    def __init__(self, in_channels):
        super(CMLSparse, self).__init__()
        self.dense = False
        self.conv3d_1 = ME.MinkowskiConvolution(
            in_channels, in_channels, 3, (2, 1, 1), dimension=3, expand_coordinates=False)
        self.conv3d_2 = ME.MinkowskiConvolution(
            in_channels, in_channels, 3, (2, 1, 1), dimension=3, expand_coordinates=False)
        self.conv3d_3 = ME.MinkowskiConvolution(
            in_channels, in_channels, 3, (2, 1, 1), dimension=3, expand_coordinates=False)
        self.out_strides = nn.Parameter(torch.Tensor([8, 1, 1]))

    def forward(self, feats, coords):
        x = ME.SparseTensor(features=feats, coordinates=coords)
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.conv3d_3(x)

        feats_out = x.F
        coords_out = x.C
        coords_out[:, 1:] = coords_out[:, 1:] / self.out_strides
        return feats_out, coords_out


