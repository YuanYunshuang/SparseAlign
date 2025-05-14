import warnings, copy
from typing import List, Optional

import torch
from torch import nn
import torch.utils.checkpoint as cp

from cosense3d.modules.utils import build_torch_module
from cosense3d.modules.utils.norm import build_norm_layer
from cosense3d.modules.utils.init import xavier_init
try:
    from cosense3d.modules.plugin.flash_attn import FlashMHA
except:
    try:
        from cosense3d.modules.plugin.flash_attn_new import FlashMHA
    except:
        FlashMHA = None
from cosense3d.modules.utils.amp import auto_fp16


def build_module(cfg):
    cfg_ = copy.deepcopy(cfg)
    attn_typ = cfg_.pop('type')
    return globals()[attn_typ](**cfg_)


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.
    """

    def __init__(self,
                 embed_dims: int,
                 feedforward_channels: int,
                 num_fcs: int=2,
                 act_cfg: dict=dict(type='ReLU', inplace=True),
                 dropout: float=0.0,
                 add_residual: bool=True):
        """

        :param embed_dims: The feature dimension. Same as
            `MultiheadAttention`.
        :param feedforward_channels: The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Defaluts to 2.
        :param num_fcs: number of fully connected layers.
        :param act_cfg: activation config.
        :param dropout: Probability of an element to be
            zeroed. Default 0.0.
        :param add_residual: Add resudual connection.
            Defaults to True.
        """
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = build_torch_module(act_cfg)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(embed_dims={self.embed_dims}, '
        repr_str += f'feedforward_channels={self.feedforward_channels}, '
        repr_str += f'num_fcs={self.num_fcs}, '
        repr_str += f'act_cfg={self.act_cfg}, '
        repr_str += f'dropout={self.dropout}, '
        repr_str += f'add_residual={self.add_residual})'
        return repr_str


class MultiheadFlashAttention(nn.Module):
    r"""A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 attn_drop: float=0.,
                 proj_drop: float=0.,
                 dropout: float=None,
                 batch_first: bool=True,
                 cache_attn_weights: bool=False,
                 **kwargs):
        """
        :param embed_dims: The embedding dimension.
        :param num_heads: Parallel attention heads.
        :param attn_drop: A Dropout layer on attn_output_weights. Default: 0.0.
        :param proj_drop: A Dropout layer after `nn.MultiheadAttention`. Default: 0.0.
        :param dropout: united dropout for both attention and projection layer.
        :param batch_first: When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
        :param cache_attn_weights: whether to cache the intermediate attention weights.
        :param kwargs:
        """
        super(MultiheadFlashAttention, self).__init__()
        if dropout is not None:
            attn_drop = dropout
            proj_drop = dropout

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = True
        self.cache_attn_weights = cache_attn_weights
        self.attn_weights = None

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                             **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Dropout(attn_drop)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward function for `MultiheadAttention`.

        :param query: The input query with shape [num_queries, bs, embed_dims] if self.batch_first is False,
            else [bs, num_queries embed_dims].
        :param key: The key tensor with shape [num_keys, bs, embed_dims] if self.batch_first is False, else
            [bs, num_keys, embed_dims]. If None, the ``query`` will be used. Defaults to None.
        :param value: The value tensor with same shape as `key`. Same in `nn.MultiheadAttention.forward`.
            Defaults to None. If None, the `key` will be used.
        :param identity: This tensor, with the same shape as x, will be used for the identity link.
            If None, `x` will be used. Defaults to None.
        :param query_pos: The positional encoding for query, with the same shape as `x`. If not None, it will
            be added to `x` before forward function. Defaults to None.
        :param key_pos: The positional encoding for `key`, with the same shape as `key`. Defaults to None.
            If not None, it will be added to `key` before forward function. If None, and `query_pos` has the same
            shape as `key`, then `query_pos` will be used for `key_pos`. Defaults to None.
        :param attn_mask: ByteTensor mask with shape [num_queries, num_keys].
            Same in `nn.MultiheadAttention.forward`. Defaults to None.
        :param key_padding_mask: ByteTensor with shape [bs, num_keys]. Defaults to None.
        :param kwargs: allow passing a more general data flow when combining with
            other operations in `transformerlayer`.
        :return: forwarded results with shape [num_queries, bs, embed_dims] if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # flash attention only support f16
            out, attn_weights = self.attn(
                q=query,
                k=key,
                v=value,
                key_padding_mask=None)

        if self.cache_attn_weights:
            self.attn_weights = attn_weights

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class MultiHeadAttentionWrapper(nn.MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super(MultiHeadAttentionWrapper, self).__init__(*args, **kwargs)
        self.fp16_enabled = True

    @auto_fp16(out_fp32=True)
    def forward_fp16(self, *args, **kwargs):
        return super(MultiHeadAttentionWrapper, self).forward(*args, **kwargs)

    def forward_fp32(self, *args, **kwargs):
        return super(MultiHeadAttentionWrapper, self).forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if self.fp16_enabled and self.training:
            return self.forward_fp16(*args, **kwargs)
        else:
            return self.forward_fp32(*args, **kwargs)


class MultiheadAttention(nn.Module):
    r"""A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    """

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 dropout: float=0.1,
                 batch_first: bool=False,
                 cache_attn_weights: bool=False,
                 fp16: bool=False,
                 **kwargs):
        """
        :param embed_dims: The embedding dimension.
        :param num_heads: Parallel attention heads.
        :param dropout: probability of Dropout layer, Default: 0.0.
        :param batch_first: When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
            Default to False.
        :param cache_attn_weights: whether to cache attention weights.
        :param fp16: whether set precision to float16
        :param kwargs:
        """
        super(MultiheadAttention, self).__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.cache_attn_weights = cache_attn_weights
        self.attn_weights = None
        self.fp16_enabled = fp16
        if fp16:
            self.attn = MultiHeadAttentionWrapper(embed_dims, num_heads, dropout,  **kwargs)
        else:
            self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout,  **kwargs)

        self.proj_drop = nn.Dropout(dropout)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward function for `MultiheadAttention`.

        :param query: The input query with shape [num_queries, bs, embed_dims] if self.batch_first is False,
            else [bs, num_queries embed_dims].
        :param key: The key tensor with shape [num_keys, bs, embed_dims] if self.batch_first is False,
            else [bs, num_keys, embed_dims]. If None, the ``query`` will be used. Defaults to None.
        :param value: The value tensor with same shape as `key`. Same in `nn.MultiheadAttention.forward`.
            Defaults to None. If None, the `key` will be used.
        :param identity: This tensor, with the same shape as x, will be used for the identity link.
            If None, `x` will be used. Defaults to None.
        :param query_pos: The positional encoding for query, with the same shape as `x`.
            If not None, it will be added to `x` before forward function. Defaults to None.
        :param key_pos: The positional encoding for `key`, with the same shape as `key`.
            Defaults to None. If not None, it will be added to `key` before `query_pos` has the same shape as `key`,
            then `query_pos` will be used for `key_pos`. Defaults to None.
        :param attn_mask: ByteTensor mask with shape [num_queries, num_keys].
            Same in `nn.MultiheadAttention.forward`. Defaults to None.
        :param key_padding_mask: ByteTensor with shape [bs, num_keys]. Defaults to None.
        :param kwargs: allow passing a more general data flow when combining with other operations in `transformerlayer`.
        :return: forwarded results with shape [num_queries, bs, embed_dims] if self.batch_first is False,
            else[bs, num_queries embed_dims].

        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1).contiguous()
            key = key.transpose(0, 1).contiguous()
            value = value.transpose(0, 1).contiguous()

        out, attn_weights = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)
        if self.batch_first:
            out = out.transpose(0, 1).contiguous()

        if self.cache_attn_weights:
            self.attn_weights = attn_weights

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=None,
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 batch_first=False,
                 with_cp=True,
                 **kwargs):
        super().__init__()
        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        num_attn = operation_order.count('self_attn') + operation_order.count('cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.batch_first = batch_first
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.use_checkpoint = with_cp

        self._init_layers(operation_order, attn_cfgs, ffn_cfgs, norm_cfg)

    def _init_layers(self, operation_order, attn_cfgs, ffn_cfgs, norm_cfg):
        self.attentions = nn.ModuleList()
        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_module(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(build_module(ffn_cfgs[ffn_index]))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def _forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks: List[torch.Tensor]=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """
        Forward function for `TransformerDecoderLayer`.

        :param query: The input query with shape [num_queries, bs, embed_dims] if self.batch_first is False,
            else [bs, num_queries embed_dims].
        :param key: The key tensor with shape [num_keys, bs, embed_dims] if self.batch_first is False,
            else [bs, num_keys, embed_dims].
        :param value: The value tensor with same shape as `key`.
        :param query_pos: The positional encoding for `query`. Default: None.
        :param key_pos: The positional encoding for `key`. Default: None.
        :param temp_memory: 2D Tensor used in calculation of corresponding attention. The length of it should equal
            to the number of `attention` in `operation_order`. Default: None.
        :param temp_pos:
        :param attn_masks: 2D Tensor used in calculation of corresponding attention. The length of it should equal
            to the number of `attention` in `operation_order`. Default: None.
        :param query_key_padding_mask: ByteTensor for `query`, with shape [bs, num_queries]. Only used in `self_attn`
            layer. Defaults to None.
        :param key_padding_mask: ByteTensor for `query`, with shape [bs, num_keys]. Default: None.
        :param kwargs: contains some specific arguments of attentions.
        :return: forwarded results with shape [num_queries, bs, embed_dims].
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=0)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=0)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                temp_memory=None,
                temp_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        :returns: Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                temp_memory,
                temp_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                )
        else:
            x = self._forward(
            query,
            key,
            value,
            query_pos,
            key_pos,
            temp_memory,
            temp_pos,
            attn_masks,
            query_key_padding_mask,
            key_padding_mask,
        )
        return x


class TransformerLayerSequence(nn.Module):
    """
    Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.
    """

    def __init__(self, transformerlayers=None, num_layers=None):
        """
        :param transformerlayers: (list[obj:`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict`)
            Config of transformerlayer in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
            it would be repeated `num_layer` times to a list[`mmcv.ConfigDict`]. Default: None.
        :param num_layers: The number of `TransformerLayer`. Default: None.
        """
        super().__init__()
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_module(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        :param query:  (Tensor) Input query with shape `(num_queries, bs, embed_dims)`.
        :param key: (Tensor) The key tensor with shape `(num_keys, bs, embed_dims)`.
        :param value: (Tensor) The value tensor with shape `(num_keys, bs, embed_dims)`.
        :param query_pos: (Tensor) The positional encoding for `query`. Default: None.
        :param key_pos: (Tensor) The positional encoding for `key`. Default: None.
        :param attn_masks: (List[Tensor], optional) Each element is 2D Tensor which is
            used in calculation of corresponding attention in operation_order. Default: None.
        :param query_key_padding_mask:  (Tensor) ByteTensor for `query`, with shape [bs, num_queries].
            Only used in self-attention Default: None.
        :param key_padding_mask:  (Tensor) ByteTensor for `query`, with shape [bs, num_keys]. Default: None.

        :returns: results with shape [num_queries, bs, embed_dims].
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query


class TransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer."""

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):
        """
        :param args:
        :param post_norm_cfg: Config of last normalization layer. Default： `LN`.
        :param return_intermediate: Whether to return intermediate outputs.
        :param kwargs:
        """

        super(TransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, **kwargs):
        """Forward function for `TransformerDecoder`.

        :param query:  (Tensor) Input query with shape `(num_query, bs, embed_dims)`.
        :return:Tensor: Results with shape [1, num_query, bs, embed_dims] when
            return_intermediate is `False`, otherwise it has shape [num_layers, num_query, bs, embed_dims].
        """
        if not self.return_intermediate:
            x = super().forward(query, *args, **kwargs)
            if self.post_norm:
                x = self.post_norm(x)[None]
            return x

        intermediate = []
        for layer in self.layers:
            query = layer(query, *args, **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        # if torch.isnan(query).any():
        #     print('TransfromerDecoder: Found nan in query.')
        # if torch.isnan(intermediate[-1]).any():
        #     print('TransfromerDecoder: Found nan in intermediate result.')
        return torch.stack(intermediate)


class PETRTemporalTransformer_(nn.Module):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(self, encoder=None, decoder=None, cross=False):
        """

        :param encoder: (`mmcv.ConfigDict` | Dict) Config of
            TransformerEncoder. Defaults to None.
        :param decoder: ((`mmcv.ConfigDict` | Dict) Config of
            TransformerDecoder. Defaults to None.
        :param cross: whether to use cross-attention.
        """
        super(PETRTemporalTransformer_, self).__init__()
        if encoder is not None:
            self.encoder = build_module(encoder)
        else:
            self.encoder = None
        self.decoder = build_module(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, memory, tgt, query_pos, pos_embed, attn_masks,
                temp_memory=None, temp_pos=None,
                mask=None, query_mask=None, reg_branch=None):
        """Forward function for `Transformer`.
        """
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()

        n, bs, c = memory.shape

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos = temp_pos.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        if not isinstance(attn_masks, list):
            attn_masks = [attn_masks, None]
        out_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            query_key_padding_mask=query_mask,
            key_padding_mask=mask,
            attn_masks=attn_masks,
            reg_branch=reg_branch,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        memory = memory.reshape(-1, bs, c).transpose(0, 1).contiguous()
        return out_dec, memory


class PETRTransformer(nn.Module):
    """
    Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
    * positional encodings are passed in MultiheadAttention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(self, encoder=None, decoder=None, cross=False):
        super(PETRTransformer, self).__init__()
        if encoder is not None:
            self.encoder = build_module(encoder)
        else:
            self.encoder = None
        self.decoder = build_module(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, memory, tgt, query_pos, pos_embed, attn_masks=None,
                mask=None, query_mask=None):
        """Forward function for `Transformer`.
        """
        memory = memory.transpose(0, 1).contiguous()
        query_pos = query_pos.transpose(0, 1).contiguous()
        pos_embed = pos_embed.transpose(0, 1).contiguous()

        n, bs, c = memory.shape

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        if not isinstance(attn_masks, list):
            attn_masks = [attn_masks]
        assert len(attn_masks) == self.decoder.layers[0].num_attn
        out_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            query_key_padding_mask=query_mask,
            key_padding_mask=mask,
            attn_masks=attn_masks,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        memory = memory.reshape(-1, bs, c).transpose(0, 1).contiguous()
        return out_dec, memory


class PETRTemporalTransformer(nn.Module):
    r"""
    Implements the DETR transformer.
    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:
    * positional encodings are passed in MultiheadAttention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    """

    def __init__(self, encoder=None, decoder=None, cross=False):
        super(PETRTemporalTransformer, self).__init__()
        if encoder is not None:
            self.encoder = build_module(encoder)
        else:
            self.encoder = None
        self.decoder = build_module(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, memory, tgt, query_pos, pos_embed, attn_masks, temp_memory=None, temp_pos=None,
                mask=None, query_mask=None, reg_branch=None):
        """Forward function for `Transformer`.
        """
        query_pos = query_pos.transpose(0, 1).contiguous()
        if memory is not None:
            memory = memory.transpose(0, 1).contiguous()
            n, bs, c = memory.shape
        if pos_embed is not None:
            pos_embed = pos_embed.transpose(0, 1).contiguous()

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos = temp_pos.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]
        if not isinstance(attn_masks, list):
            attn_masks = [attn_masks]
        assert len(attn_masks) == self.decoder.layers[0].num_attn
        out_dec = self.decoder(
            query=tgt,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            query_key_padding_mask=query_mask,
            key_padding_mask=mask,
            attn_masks=attn_masks,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        if memory is not None:
            memory = memory.reshape(-1, bs, c).transpose(0, 1).contiguous()
        return out_dec, memory