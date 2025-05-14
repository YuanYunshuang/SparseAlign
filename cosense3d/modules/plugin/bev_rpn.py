import torch
from torch import nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, k, s, p, activation=True,
                 batch_norm=True):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation:
            return F.relu(x, inplace=True)
        else:
            return x


class RPN(nn.Module):
    def __init__(self, anchor_num=2):
        super(RPN, self).__init__()
        self.anchor_num = anchor_num

        self.block_1 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_1 += [Conv2d(128, 128, 3, 1, 1) for _ in range(3)]
        self.block_1 = nn.Sequential(*self.block_1)

        self.block_2 = [Conv2d(128, 128, 3, 2, 1)]
        self.block_2 += [Conv2d(128, 128, 3, 1, 1) for _ in range(5)]
        self.block_2 = nn.Sequential(*self.block_2)

        self.block_3 = [Conv2d(128, 256, 3, 2, 1)]
        self.block_3 += [nn.Conv2d(256, 256, 3, 1, 1) for _ in range(5)]
        self.block_3 = nn.Sequential(*self.block_3)

        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, 4, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(128, 256, 2, 2, 0),
                                      nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(128, 256, 1, 1, 0),
                                      nn.BatchNorm2d(256))

    def forward(self, x):
        x = self.block_1(x)
        x_skip_1 = x
        x = self.block_2(x)
        x_skip_2 = x
        x = self.block_3(x)
        x_0 = self.deconv_1(x)
        x_1 = self.deconv_2(x_skip_2)
        x_2 = self.deconv_3(x_skip_1)
        x = torch.cat((x_0, x_1, x_2), 1)
        return x


class CustomRPN(nn.Module):
    def __init__(self, strides=[2, 2, 2], down_sample=2, num_layers=3, in_channels=128, out_channels=256):
        super(CustomRPN, self).__init__()
        self.strides = strides
        mid_channels = in_channels * 2
        self.n_blocks = len(strides)
        up_stride = 1

        for i, s in enumerate(self.strides):
            channels = mid_channels if i == self.n_blocks - 1 else in_channels
            block = [Conv2d(in_channels, channels, 3, s, 1)]
            block += [Conv2d(channels, channels, 3, 1, 1) for _ in range(num_layers)]
            setattr(self, f'block_{i + 1}', nn.Sequential(*block))
            up_stride *= s
            stride = up_stride // down_sample
            setattr(self, f'deconv_{self.n_blocks  - i}',
                    nn.Sequential(nn.ConvTranspose2d(channels, mid_channels, stride, stride, 0),
                                  nn.BatchNorm2d(mid_channels))
                    )
        self.out_conv = nn.Sequential(nn.ConvTranspose2d(mid_channels * 3, out_channels, 1, 1, 0),
                                      nn.BatchNorm2d(out_channels))

    def forward(self, x):
        ret_dict = {}
        down_stride = 1
        for i, s in enumerate(self.strides):
            x = getattr(self, f'block_{i + 1}')(x)
            down_stride *= s
            ret_dict[f'p{down_stride}'] = x

        out = []
        for i, s in enumerate(self.strides):
            x = getattr(self, f'deconv_{i + 1}')(ret_dict[f'p{down_stride}'])
            down_stride = down_stride // s
            out.append(x)
        out = self.out_conv(torch.cat(out, 1))

        return out, ret_dict