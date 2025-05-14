# Copyright (c) OpenMMLab. All rights reserved. Modified by Yunshuang Yuan.
import inspect
from typing import Dict, Tuple, Union
from importlib import import_module

import torch.nn as nn
import re  # type: ignore


def infer_abbr(class_type: type) -> str:
    """Infer abbreviation from the class name.

    This method will infer the abbreviation to map class types to
    abbreviations.

    Rule 1: If the class has the property "abbr", return the property.
    Rule 2: Otherwise, the abbreviation falls back to snake case of class
    name, e.g. the abbreviation of ``FancyBlock`` will be ``fancy_block``.

    :param class_type:  The norm layer type.
    :return: The inferred abbreviation.
    """

    def camel2snack(word):
        """Convert camel case word into snack case.

        Modified from `inflection lib
        <https://inflection.readthedocs.io/en/latest/#inflection.underscore>`_.

        Example::

            >>> camel2snack("FancyBlock")
            'fancy_block'
        """

        word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
        word = word.replace('-', '_')
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_  # type: ignore
    else:
        return camel2snack(class_type.__name__)


def build_plugin_layer(cfg: Dict,
                       postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build plugin layer.

    :param cfg: cfg should contain:

            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
    :param postfix: appended into norm abbreviation to
            create named layer. Default: ''.
    :param kwargs:
    :return: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    try:
        pkg, cls = layer_type.rsplit('.', 1)
        plugin_layer = import_module(pkg).get(cls)
    except:
        raise KeyError(f'Unrecognized plugin type {layer_type}')

    abbr = infer_abbr(plugin_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    layer = plugin_layer(**kwargs, **cfg_)

    return name, layer


def build_plugin_module(cfg: Dict):
    cfg_ = cfg.copy()
    type_ = cfg_.pop('type')
    module_name, cls_name = type_.split('.')
    module = import_module(f'{__package__}.{module_name}')
    cls_inst = getattr(module, cls_name)(**cfg_)
    return cls_inst