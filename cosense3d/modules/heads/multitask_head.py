from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules import build_module
from cosense3d.modules.plugin import build_plugin_module


class MultiTaskHead(BaseModule):
    def __init__(self,
                 heads,
                 strides,
                 losses,
                 formatting=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        modules = []
        gt_keys = set(self.gt_keys)
        for i, h in enumerate(heads):
            if len(h.get('gt_keys', [])) == 0:
                cur_gt_keys = self.gt_keys
            else:
                cur_gt_keys = h['gt_keys']
                gt_keys.update(set(cur_gt_keys))
            h.update(dict(
                stride=strides[i],
                gather_keys=self.gather_keys,
                scatter_keys=[self.scatter_keys[i]],
                gt_keys=cur_gt_keys,
            ))
            modules.append(build_module(h))
        self.heads = nn.ModuleList(modules)
        self.gt_keys = list(gt_keys)
        if formatting is None:
            self.formatting = [None] * len(self.heads)
        else:
            assert len(formatting) == len(self.heads)
            self.formatting = []
            for fmt in formatting:
                self.formatting.append(build_plugin_module(fmt))

    def forward(self, tensor_list, *args, **kwargs):
        out = {}
        for i, h in enumerate(self.heads):
            x = h(tensor_list, *args, **kwargs)
            if self.formatting[i] is not None:
                for k, v in x.items():
                    x[k] = self.formatting[i](x[k])
            out.update(x)

        return out

    def loss(self, *args, **kwargs):
        kl = len(self.scatter_keys)
        heads_out = args[:kl]
        gt_dict = {k:args[kl+i] for i, k in enumerate(self.gt_keys)}
        loss_dict = {}
        for i, h in enumerate(self.heads):
            if self.losses[i]:
                gt_list = [gt_dict[k] for k in h.gt_keys]
                loss_dict.update(h.loss(heads_out[i], *gt_list, **kwargs))
        return loss_dict