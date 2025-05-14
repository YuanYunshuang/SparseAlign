import torch
from torch import nn
from einops import rearrange, repeat, reduce
from torchvision.models.resnet import Bottleneck

from cosense3d.modules.plugin.cobevt import CrossViewSwapAttention, Attention, BEVEmbedding
from cosense3d.modules import BaseModule
ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


class FAXModule(BaseModule):
    def __init__(
            self,
            middle,
            dim,
            img_size,
            strides,
            feat_dims,
            cross_view,
            cross_view_swap,
            bev_embedding,
            self_attn,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.img_size = img_size

        cross_views = list()
        layers = list()
        downsample_layers = list()

        for i, (stride, num_layers) in enumerate(zip(strides, middle)):
            feat_dim = feat_dims[i]
            feat_height, feat_width = img_size[0] // stride, img_size[1] // stride

            cva = CrossViewSwapAttention(feat_height, feat_width, feat_dim,
                                         dim[i], i,
                                         **cross_view, **cross_view_swap)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim[i]) for _ in range(num_layers)])
            layers.append(layer)

            if i < len(middle) - 1:
                downsample_layers.append(nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(dim[i], dim[i] // 4,
                                  kernel_size=3, stride=1,
                                  padding=1, bias=False),
                        nn.PixelUnshuffle(2),
                        nn.Conv2d(dim[i+1], dim[i+1],
                                  3, padding=1, bias=False),
                        nn.BatchNorm2d(dim[i+1]),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim[i+1],
                                  dim[i+1], 1, padding=0, bias=False),
                        nn.BatchNorm2d(dim[i+1])
                        )))

        self.bev_embedding = BEVEmbedding(dim[0], **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.self_attn = Attention(dim[-1], **self_attn)

    def forward(self, img_feat, intrinsic, extrinsic, **kwargs):
        B = len(img_feat)
        N = len(intrinsic[0])
        intrinsic = self.cat_list(intrinsic, recursive=True)
        extrinsic = self.cat_list(extrinsic, recursive=True)
        I_inv = torch.stack([I.inverse()[:3, :3] for I in intrinsic], dim=0
                            ).reshape(B, N, 3, 3)
        E_inv = torch.stack([E.inverse() for E in extrinsic], dim=0
                            ).reshape(B, N, 4, 4)

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=B)  # B d H W

        for i, (cross_view, layer) in enumerate(zip(self.cross_views, self.layers)):
            feature = torch.stack([feat[i] for feat in img_feat], dim=0)

            x = cross_view(i, x, self.bev_embedding, feature, I_inv, E_inv)
            x = layer(x)
            if i < len(img_feat[0])-1:
                x = self.downsample_layers[i](x)

        x = self.self_attn(x)
        return {self.scatter_keys[0]: x}

