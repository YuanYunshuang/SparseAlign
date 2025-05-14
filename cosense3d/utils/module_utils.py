import copy
import warnings
from importlib import import_module
from packaging.version import parse
from torch import nn


def build_norm_layer(cfgs, shape):
    if cfgs['type'] == 'LN':
        _cfgs = copy.copy(cfgs)
        _cfgs.pop('type')
        norm = nn.LayerNorm(shape, **_cfgs)
    else:
        raise NotImplementedError
    return norm


def build_dropout(cfgs):
    if cfgs['type'] == 'Dropout':
        dropout = nn.Dropout(cfgs['drop_prob'])
    else:
        raise NotImplementedError
    return dropout


def get_target_module(target):
    module, cls_name = target.rsplit('.', 1)
    module = import_module(module)
    cls_obj = getattr(module, cls_name)
    return cls_obj


def instantiate_target_module(target, cfg=None, **kwargs):
    if cfg is not None:
        return get_target_module(target)(cfg)
    else:
        return get_target_module(target)(**kwargs)


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)