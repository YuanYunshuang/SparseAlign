"""
This class is about swap fusion applications
"""
import torch
from einops import rearrange
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin.cobevt import NaiveDecoder


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# swap attention -> max_vit
class Attention(nn.Module):
    """
    Unit Attention class. Todo: mask is not added yet.

    Parameters
    ----------
    dim: int
        Input feature dimension.
    dim_head: int
        The head dimension.
    dropout: float
        Dropout rate
    agent_size: int
        The agent can be different views, timestamps or vehicles.
    """

    def __init__(
            self,
            dim,
            dim_head=32,
            dropout=0.,
            agent_size=6,
            window_size=7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, \
            'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.window_size = [agent_size, window_size, window_size]

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        self.relative_position_bias_table = nn.Embedding(
            (2 * self.window_size[0] - 1) *
            (2 * self.window_size[1] - 1) *
            (2 * self.window_size[2] - 1),
            self.heads)  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for
        # each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # 3, Wd, Wh, Ww
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww

        # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = \
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= \
            (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, mask=None):
        # x shape: b, l, h, w, w_h, w_w, c
        batch, agent_size, height, width, window_height, window_width, _, device, h \
            = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')
        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      (q, k, v))
        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias
        L = agent_size * window_height * window_width
        bias = self.relative_position_bias_table(self.relative_position_index[:L, :L])
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # mask shape if exist: b x y w1 w2 e l
        if mask is not None:
            # b x y w1 w2 e l -> (b x y) 1 (l w1 w2)
            mask = rearrange(mask, 'b x y w1 w2 e l -> (b x y) e (l w1 w2)')
            # (b x y) 1 1 (l w1 w2) = b h 1 n
            mask = mask.unsqueeze(1)
            sim = sim.masked_fill(mask == 0, -float('inf'))

        # attention
        attn = self.attend(sim)
        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # merge heads
        out = rearrange(out, 'b h (l w1 w2) d -> b l w1 w2 (h d)',
                        l=agent_size, w1=window_height, w2=window_width)

        # combine heads out
        out = self.to_out(out)
        return rearrange(out, '(b x y) l w1 w2 d -> b l x y w1 w2 d',
                         b=batch, x=height, y=width)


class SwapFusionBlockMask(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention with
    mask enabled for multi-vehicle cooperation.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(SwapFusionBlockMask, self).__init__()

        self.window_size = window_size

        self.window_attention = PreNormResidual(input_dim,
                                                Attention(input_dim, dim_head,
                                                          drop_out,
                                                          agent_size,
                                                          window_size))
        self.window_ffd = PreNormResidual(input_dim,
                                          FeedForward(input_dim, mlp_dim,
                                                      drop_out))
        self.grid_attention = PreNormResidual(input_dim,
                                              Attention(input_dim, dim_head,
                                                        drop_out,
                                                        agent_size,
                                                        window_size))
        self.grid_ffd = PreNormResidual(input_dim,
                                        FeedForward(input_dim, mlp_dim,
                                                    drop_out))

    def forward(self, x, mask):
        # x: b l c h w
        # mask: b h w 1 l
        # window attention -> grid attention
        mask_swap = mask

        # mask b h w 1 l -> b x y w1 w2 1 L
        mask_swap = rearrange(mask_swap,
                              'b (x w1) (y w2) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.window_attention(x, mask=mask_swap)
        x = self.window_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (x w1) (y w2)')

        # grid attention
        mask_swap = mask
        mask_swap = rearrange(mask_swap,
                              'b (w1 x) (w2 y) e l -> b x y w1 w2 e l',
                              w1=self.window_size, w2=self.window_size)
        x = rearrange(x, 'b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=self.window_size, w2=self.window_size)
        x = self.grid_attention(x, mask=mask_swap)
        x = self.grid_ffd(x)
        x = rearrange(x, 'b m x y w1 w2 d -> b m d (w1 x) (w2 y)')

        return x


class SwapFusionBlock(nn.Module):
    """
    Swap Fusion Block contains window attention and grid attention.
    """

    def __init__(self,
                 input_dim,
                 mlp_dim,
                 dim_head,
                 window_size,
                 agent_size,
                 drop_out):
        super(SwapFusionBlock, self).__init__()
        # b = batch * max_cav
        self.block = nn.Sequential(
            Rearrange('b m d (x w1) (y w2) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (x w1) (y w2)'),

            Rearrange('b m d (w1 x) (w2 y) -> b m x y w1 w2 d',
                      w1=window_size, w2=window_size),
            PreNormResidual(input_dim, Attention(input_dim, dim_head, drop_out,
                                                 agent_size, window_size)),
            PreNormResidual(input_dim,
                            FeedForward(input_dim, mlp_dim, drop_out)),
            Rearrange('b m x y w1 w2 d -> b m d (w1 x) (w2 y)'),
        )

    def forward(self, x, mask=None):
        # todo: add mask operation later for mulit-agents
        x = self.block(x)
        return x


class SwapFusionEncoder(BaseModule):
    """
    Data rearrange -> swap block -> mlp_head
    """

    def __init__(self,
                 input_dim=128,
                 mlp_dim=256,
                 agent_size=5,
                 window_size=8,
                 dim_head=32,
                 drop_out=0.1,
                 depth=3,
                 mask=False,
                 decoder=None,
                 **kwargs):
        super(SwapFusionEncoder, self).__init__(**kwargs)
        self.layers = nn.ModuleList([])
        self.depth = depth
        self.mask = mask
        swap_fusion_block = SwapFusionBlockMask if self.mask else SwapFusionBlock

        for i in range(self.depth):
            block = swap_fusion_block(input_dim,
                                      mlp_dim,
                                      dim_head,
                                      window_size,
                                      agent_size,
                                      drop_out)
            self.layers.append(block)

        # mlp head
        self.mlp_head = nn.Sequential(
            Reduce('b m d h w -> b d h w', 'mean'),
            Rearrange('b d h w -> b h w d'),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange('b h w d -> b d h w')
        )

        if decoder is not None:
            self.decoder = NaiveDecoder(decoder)

    def forward(self, ego_feat, coop_cpm, **kwargs):
        B = len(ego_feat)
        C, H, W = ego_feat[0].shape
        x = []
        mask = []
        num_cavs = []
        for xe, xc in zip(ego_feat, coop_cpm):
            values = xc.values()
            ego_mask = torch.ones_like(xe[:1])
            x.append([xe,] + [v['bev_feat'] for v in values])
            mask.append([ego_mask,] + [v['bev_mask'] for v in values])
            num_cavs.append(len(values) + 1)
        l = max(num_cavs)
        x_pad = ego_feat[0].new_zeros(B, l, C, H, W)
        mask_pad = ego_feat[0].new_zeros(B, H, W, 1, l)
        for i in range(B):
            x_pad[i, :len(x[i])] = torch.stack(x[i], dim=0)
            mask_pad[i, :, :, :, :len(x[i])] = torch.stack(mask[i], dim=-1).permute(1, 2, 0, 3)
        for stage in self.layers:
            x_pad = stage(x_pad, mask=mask_pad)
        out = self.mlp_head(x_pad)

        if hasattr(self, 'decoder'):
            out = self.decoder(out.unsqueeze(1))
            out = rearrange(out, 'b l c h w -> (b l) c h w')
        return {self.scatter_keys[0]: out}


if __name__ == "__main__":
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = {'input_dim': 512,
            'mlp_dim': 512,
            'agent_size': 4,
            'window_size': 8,
            'dim_head': 4,
            'drop_out': 0.1,
            'depth': 2,
            'mask': True
            }
    block = SwapFusionEncoder(args)
    block.cuda()
    test_data = torch.rand(1, 4, 512, 32, 32)
    test_data = test_data.cuda()
    mask = torch.ones(1, 32, 32, 1, 4)
    mask = mask.cuda()

    output = block(test_data, mask)
    print(output)
