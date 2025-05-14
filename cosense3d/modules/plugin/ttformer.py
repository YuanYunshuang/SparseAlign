
from cosense3d.modules.plugin.transformer import *


class TemporalTransformer(nn.Module):
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
        super(TemporalTransformer, self).__init__()
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

    def forward(self, tgt, query_pos, attn_mask=None,
                temp_memory=None, temp_pos=None,
                mask=None, query_mask=None, reg_branch=None):
        """Forward function for `Transformer`.
        """
        query_pos = query_pos.transpose(0, 1).contiguous()
        tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos = temp_pos.transpose(0, 1).contiguous()

        # out_dec: [num_layers, num_query, bs, dim]

        attn_masks = [attn_mask]
        assert len(attn_masks) == self.decoder.layers[0].num_attn
        out_dec = self.decoder(
            query=tgt,
            key=None,
            value=None,
            key_pos=None,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            query_key_padding_mask=query_mask,
            key_padding_mask=mask,
            attn_masks=attn_masks,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        return out_dec