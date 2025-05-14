# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
#  Modified by Yunshuang Yuan
# ------------------------------------------------------------------------
# flash-attention
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import (
    xavier_uniform_,
    constant_,
    xavier_normal_
)
from torch.nn.functional import linear

from einops import rearrange


from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func, _get_block_size
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis
from cosense3d.modules.utils.test_flash_attn import convert_flash_attn_S_to_softmax, \
    generate_random_padding_mask



def flash_attn_unpadded_kvpacked_test(q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk, dropout_p, softmax_scale,
                                      causal, batch_size):
    d = q.shape[-1]
    device = q.device
    output_unpad, sm_lse, S_dmask = flash_attn_unpadded_kvpacked_func(
        q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
        dropout_p, return_attn_probs=True, causal=causal, softmax_scale=softmax_scale
    )
    query_padding_mask = generate_random_padding_mask(max_sq, batch_size, device, mode='full')
    key_padding_mask = generate_random_padding_mask(max_sk, batch_size, device, mode='full')
    S_dmask_converted = convert_flash_attn_S_to_softmax(
        S_dmask, query_padding_mask, key_padding_mask, d, dropout_p > 0.0, causal=causal
    )
    return output_unpad, S_dmask_converted


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    """

    def __init__(self,
                 softmax_scale: float=None,
                 attention_dropout: float=0.0,
                 return_attn_weights: float=False,
                 device: str=None,
                 dtype: type=None):
        """

        :param softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        :param attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        :param return_attn_weights:
        :param device:
        :param dtype:
        """
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True
        self.return_attn_weights = return_attn_weights

    def forward(self,
                q: torch.Tensor,
                kv: torch.Tensor,
                causal: bool=False,
                key_padding_mask: torch.Tensor=None):
        """Implements the multihead softmax attention.

        :param q: The tensor containing the query. (B, T, H, D)
        :param kv: The tensor containing the key, and value. (B, S, 2, H, D)
        :param causal:
        :param key_padding_mask: a bool tensor of shape (B, S)
        :return:
        """
        # assert q.dtype in [torch.float16, torch.bfloat16] and kv.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        assert q.shape[0] == kv.shape[0] and q.shape[-2] == kv.shape[-2] and q.shape[-1] == kv.shape[-1]

        batch_size = q.shape[0]
        seqlen_q, seqlen_k = q.shape[1], kv.shape[1]
        if key_padding_mask is None:
            q, kv = rearrange(q, 'b s ... -> (b s) ...'), rearrange(kv, 'b s ... -> (b s) ...')
            max_sq, max_sk = seqlen_q, seqlen_k
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                                        device=kv.device)
            if self.training or not self.return_attn_weights:
                output = flash_attn_unpadded_kvpacked_func(
                    q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
                attn_weights = None
            else:
                Q, K, V = q.permute(1, 0, 2), kv[:, 0].permute(1, 0, 2), kv[:, 1].permute(1, 0, 2)
                attn_weights = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
                # attn_weights = torch.dropout(attn_weights, self.dropout_p, train=False)
                output = attn_weights @ V
                attn_weights = attn_weights.mean(dim=0)
                output = output.permute(1, 0, 2)

                # output, attn_weights = flash_attn_unpadded_kvpacked_test(
                #     q, kv, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                #     self.dropout_p if self.training else 0.0,
                #     softmax_scale=self.softmax_scale, causal=causal, batch_size=batch_size
                # )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
                attn_weights = rearrange(attn_weights, '(b s) ... -> b s ...', b=batch_size)
                # attn_weights = attn_weights.mean(dim=1)
        else:
            nheads = kv.shape[-2]
            q = rearrange(q, 'b s ... -> (b s) ...')
            max_sq = seqlen_q
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                        device=q.device)
            x = rearrange(kv, 'b s two h d -> b s (two h d)')
            x_unpad, indices, cu_seqlens_k, max_sk = unpad_input(x, key_padding_mask)
            x_unpad = rearrange(x_unpad, 'nnz (two h d) -> nnz two h d', two=2, h=nheads)
            output_unpad = flash_attn_unpadded_kvpacked_func(
                q, x_unpad, cu_seqlens_q, cu_seqlens_k, max_sq, max_sk,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )
            output = rearrange(output_unpad, '(b s) ... -> b s ...', b=batch_size)
            attn_weights = None

        return output, attn_weights


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.bias = bias

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)
        """
        # q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)
        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights
