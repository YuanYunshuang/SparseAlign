"""
Class used to downsample features by 3*3 conv
"""
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convoltuion
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: bool):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownsampleConv(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_sizes=[1],
                 dims=[256],
                 strides=[1],
                 paddings=[0]):
        super(DownsampleConv, self).__init__()
        self.layers = nn.ModuleList([])

        for ksize, dim, stride, padding in zip(
                kernel_sizes, dims, strides, paddings):
            self.layers.append(DoubleConv(in_channels,
                                          dim,
                                          kernel_size=ksize,
                                          stride=stride,
                                          padding=padding))
            in_channels = dim

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x