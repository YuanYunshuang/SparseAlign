
import math
import torch
from torch import nn

from cosense3d.modules import build_module


class ForwardRunner(nn.Module):
    def __init__(self, shared_modules, data_manager, dist=False, chunk_size=24, **kwargs):
        super().__init__()
        self.lidar_range = torch.tensor(data_manager.lidar_range)
        self.data_manager = data_manager
        self.dist = dist
        # if the fwd items of a module exits the GPU capacity, run them in several mini batches
        self.chunk_size = chunk_size

        module_dict = {}
        self.module_keys = []
        for k, v in shared_modules.items():
            if 'type' not in v:
                continue
            v['dist'] = dist
            module = build_module(v)
            if module.freeze:
                module.freeze_parameters()
            module_dict[k] = module
            self.module_keys.append(k)

        self.shared_modules = nn.ModuleDict(module_dict)

    def to_gpu(self, gpu_id):
        for n, m in self.shared_modules.items():
            sync_func = m.to_gpu(gpu_id)
            if sync_func is not None:
                self.shared_modules[n] = sync_func(m)

    def gather_cav_ids(self, tasks):
        return [t[0] for t in tasks]

    def forward(self, tasks, with_grad=True, **kwargs):
        if with_grad:
            self._forward(tasks, **kwargs)
        else:
            with torch.no_grad():
                self._forward(tasks, **kwargs)

    def _forward(self, tasks, **kwargs):
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            task_ids = self.gather_cav_ids(task_list)
            n_task = len(task_ids)
            s = self.chunk_size
            if n_task > s and 0 < n_task % s < 4:
                s = int(math.ceil(n_task / math.ceil(n_task / s)))
            chunks = [task_ids[i:i + s] for i in range(0, len(task_ids), s)]
            res = {k: [] for k in module.scatter_keys}
            for tids in chunks:
                data = self.data_manager.gather(tids, module.gather_keys)
                cur_res = module(*data, **kwargs)
                for k in module.scatter_keys:
                    res[k].extend(cur_res[k])
            self.data_manager.scatter(task_ids, res)

    def loss(self, tasks, **kwargs):
        loss_dict = {}
        loss = 0
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            if module.freeze:
                continue
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.scatter_keys + module.gt_keys)
            ldict = module.loss(*data, **kwargs)
            for k, v in ldict.items():
                prefix = task_name.replace('_head', '')
                loss_dict[f'{prefix}.{k}'] = v
                loss = loss + v
        loss_dict['total_loss'] = loss
        return loss, loss_dict

    def frame_loss(self, tasks, **kwargs):
        loss_dict = {}
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            if module.freeze:
                continue
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.scatter_keys + module.gt_keys)
            ldict = module.loss(*data, **kwargs)
            for k, v in ldict.items():
                prefix = task_name.replace('_head', '')
                loss_dict[f'{prefix}.{k}'] = v
        return loss_dict





