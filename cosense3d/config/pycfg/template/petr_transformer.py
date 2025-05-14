
def get_petr_transformer_cfg(flash_attn=True, embed_dims=256):
    return dict(
                type='transformer.PETRTemporalTransformer',
                decoder=dict(
                    type='TransformerDecoder',
                    return_intermediate=True,
                    num_layers=1,
                    transformerlayers=dict(
                        type='TransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention', #fp16 for 2080Ti training (save GPU memory).
                                embed_dims=embed_dims,
                                num_heads=8,
                                dropout=0.1,
                                fp16=False),
                            dict(
                                type='MultiheadFlashAttention' if flash_attn else 'MultiheadAttention',
                                embed_dims=embed_dims,
                                num_heads=8,
                                dropout=0.1,
                                fp16=flash_attn
                            ),
                            ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=1024,
                            num_fcs=2,
                            dropout=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        with_cp=False,  ###use checkpoint to save memory
                        operation_order=('self_attn', 'norm',
                                         'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )
            )


def get_transformer_cfg(flash_attn=True, embed_dims=256):
    return dict(
                type='ttformer.TemporalTransformer',
                decoder=dict(
                    type='TransformerDecoder',
                    return_intermediate=True,
                    num_layers=1,
                    transformerlayers=dict(
                        type='TransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention', #fp16 for 2080Ti training (save GPU memory).
                                embed_dims=embed_dims,
                                num_heads=8,
                                dropout=0.1,
                                fp16=False),
                            ],
                        ffn_cfgs=dict(
                            type='FFN',
                            embed_dims=embed_dims,
                            feedforward_channels=1024,
                            num_fcs=2,
                            dropout=0.,
                            act_cfg=dict(type='ReLU', inplace=True),
                        ),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        with_cp=False,  ###use checkpoint to save memory
                        operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                )
            )


