import copy
from torch import nn


def build_torch_module(cfg):
    cfg_ = copy.deepcopy(cfg)
    module_name = cfg_.pop('type')
    module = getattr(nn, module_name)(**cfg_)
    return module