import torch
from torch import nn
from typing import List, Dict, Optional
import importlib

from cosense3d.modules.utils.common import cat_coor_with_idx
from cosense3d.modules.utils.me_utils import ME


def build_module(module_cfg):
    module_full_path=module_cfg['type']
    package, module_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'cosense3d.modules.{package}')
    cls_obj = getattr(module, module_name, None)
    assert cls_obj is not None, f"Class '{module_name}' not found."
    try:
        inst = cls_obj(**module_cfg)
    except Exception as e:
        raise Exception(f"{module_name}:{e.__repr__()}")
    return inst


class BaseModule(nn.Module):
    def __init__(self, gather_keys, scatter_keys, gt_keys=[], freeze=False, **kwargs):
        super(BaseModule, self).__init__()
        self.gather_keys = gather_keys
        self.scatter_keys = scatter_keys
        self.gt_keys = gt_keys
        self.freeze = freeze

    def to_gpu(self, gpu_id):
        self.to(gpu_id)
        addtional_sync_func = nn.SyncBatchNorm.convert_sync_batchnorm
        return None

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """This must be implemented in head module."""
        # TODO: Create Head base module.
        pass

    def prepare_vis_data(self):
        pass

    def format_input(self, input: List):
        pass

    def format_output(self, output, B):
        pass

    def cat_data_from_list(self, input, key=None, pad_idx=False):
        if key is not None:
            data = [x[key] for x in input]
        else:
            data = input
        if isinstance(data[0], torch.Tensor):
            if pad_idx:
                return cat_coor_with_idx(data)
            else:
                return torch.cat(data, dim=0)
        else:
            return data

    def stack_data_from_list(self, input, key=None):
        if key is not None:
            data = [x[key] for x in input]
        else:
            data = input
        if isinstance(data[0], torch.Tensor):
            return torch.stack(data, dim=0)
        else:
            return data


    def cat_list(self, x_list, recursive=False):
        """Concatenate sub_lists to one list"""
        if len(x_list) > 0 and isinstance(x_list[0], list):
            out = []
            for x in x_list:
                out.extend(self.cat_list(x) if recursive else x)
            return out
        else:
            return x_list

    def cat_dict_list(self, d_list: List[Dict]):
        out_dict = {k:[] for k in d_list[0].keys()}
        for k in d_list[0].keys():
            for d in d_list:
                out_dict[k].extend(d[k])
        return out_dict

    def stack_dict_list(self, d_list: List[Dict]):
        out_dict = {k:[] for k in d_list[0].keys()}
        for k in d_list[0].keys():
            for d in d_list:
                out_dict[k].append(d[k])
            out_dict[k] = torch.stack(out_dict[k], dim=0)
        return out_dict

    def compose_imgs(self, img_list):
        imgs = [img for x in img_list for img in x]
        return torch.stack(imgs, dim=0)

    def compose_stensor(self, stensor_list, stride):
        coor = [stensor[f'p{stride}']['coor'] for stensor in stensor_list]
        coor = cat_coor_with_idx(coor)
        feat = [stensor[f'p{stride}']['feat'] for stensor in stensor_list]
        feat = torch.cat(feat, dim=0)
        if 'ctr' in stensor_list[0][f'p{stride}']:
            ctr = [stensor[f'p{stride}']['ctr'] for stensor in stensor_list]
            ctr = torch.cat(ctr, dim=0)
        else:
            ctr = None
        return coor, feat, ctr

    def decompose_stensor(self, res, N):
        # decompose batch
        for k, v in res.items():
            if isinstance(v, ME.SparseTensor):
                coor, feat = v.decomposed_coordinates_and_features
                ctr = None
            elif isinstance(v, dict):
                coor, feat, ctr = [], [], []
                for i in range(N):
                    mask = v['coor'][:, 0] == i
                    coor.append(v['coor'][mask, 1:])
                    feat.append(v['feat'][mask])
                    ctr.append(v['ctr'][mask])
            else:
                raise NotImplementedError
            res[k] = {'coor': coor, 'feat': feat, 'ctr': ctr}

        # compose result list
        res_list = self.compose_result_list(res, N)
        return res_list

    def compose_result_list(self, res, N):
        """

        :param res: dict(k:list)
        :param N:
        :return:
        """
        keys = res.keys()
        res_list = []
        for i in range(N):
            cur_res = {}
            for k, v in res.items():
                if isinstance(v, dict):
                    cur_res[k] = {
                        'coor': v['coor'][i],
                        'feat': v['feat'][i],
                        'ctr': v['ctr'][i]
                    }
                elif isinstance(v, list) or isinstance(v, torch.Tensor):
                    cur_res[k] = v[i]
                else:
                    raise NotImplementedError
            res_list.append(cur_res)
        return res_list

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gather_keys={self.gather_keys}, '
        repr_str += f'scatter_keys={self.scatter_keys})'
        return repr_str