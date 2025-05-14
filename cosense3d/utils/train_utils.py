import copy
import random
import warnings

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


def get_gpu_architecture():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_props = torch.cuda.get_device_properties(device)
        return gpu_props.major * 10 + gpu_props.minor
    else:
        return 0


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def build_optimizer(model, cfg):
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,  lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'],
                                  betas=tuple(cfg['betas']))

    return optimizer


def build_lr_scheduler(optimizer, cfg, steps_per_epoch):
    cfg_ = copy.copy(cfg)
    policy = cfg_.pop('policy', 'MultiStepLR')
    if policy == 'MultiStepLR':
        # construct a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg['milestones'],
                                                            gamma=cfg['gamma'])
    elif policy == 'CosineAnnealingWarm':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        num_steps = cfg['epochs'] * steps_per_epoch
        warmup_lr = cfg['warmup_lr']
        warmup_steps = cfg['warmup_epochs'] * steps_per_epoch
        lr_min = cfg['lr_min']

        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError

    return lr_scheduler


def is_tensor_to_cuda(data, device=0):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = is_tensor_to_cuda(v, device)
        return data
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list) or isinstance(data, tuple):
        data_t = []
        for i in range(len(data)):
            data_t.append(is_tensor_to_cuda(data[i], device))
        return data_t
    else:
        return data


def load_tensors_to_gpu(batch_dict, device=0):
    """
    Load all tensors in batch_dict to gpu
    """

    for k, v in batch_dict.items():
        batch_dict[k] = is_tensor_to_cuda(v, device=device)


def load_model_dict(model, pretrained_dict, dist=False):
    try:
        model.load_state_dict(pretrained_dict)
    except:
        UnmatchedParams = ""
        # 1. filter out unnecessary keys
        model_dict = model.state_dict()
        matched_dict = {}

        pretrained_keys = list()
        for k, v in pretrained_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            if k in model_dict and v.shape == model_dict[k].shape:
                matched_dict[k] = v
            elif k in model_dict and v.shape != model_dict[k].shape:
                UnmatchedParams += f"{k} : Unmatched shape ({v.shape} -> {model_dict[k].shape})\n"
            else:
                UnmatchedParams += f"{k} : Pretrained parameters not in model dict\n"
            pretrained_keys.append(k)
        for k in set(model_dict.keys()) - set(pretrained_keys):
            UnmatchedParams += f"{k} : Model parameters not in pretrained dict\n"
        if len(UnmatchedParams) > 0:
            warnings.warn("Model state dict does not match pretrained state dict. Unmatched parameters are:\n"
                          + UnmatchedParams)
        # 2. overwrite entries in the existing state dict
        model_dict.update(matched_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model


def clip_grads(params, max_norm=35, norm_type=2):
    params = list(
        filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        total_norm = clip_grad_norm_(params, max_norm=max_norm, norm_type=norm_type)
        return total_norm

