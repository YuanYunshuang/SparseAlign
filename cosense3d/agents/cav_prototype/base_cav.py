import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP


class BaseCAV:
    def __init__(self, id: str, mapped_id: int, is_ego: bool,
                 lidar_range: torch.Tensor, memory_len: int,
                 lidar_pose: torch.Tensor=None, require_grad: bool=False,
                 seq_len: int=1, **kwargs):
        """
        Base class for CAV prototype.

        :param id: agent id.
        :param mapped_id: remapped id.
        :param is_ego: if the agent is an ego agent.
        :param lidar_range: visible lidar range,
        :param memory_len: memory length for memory queue.
        :param lidar_pose: lidar pose in shape (4, 4).
        :param require_grad: if True, the gradients will be calculated for this agent during training.
        :param seq_len: sequence length of the input data.
        :param kwargs: additional key-value arguments.
        """
        self.id = id
        self.mapped_id = mapped_id
        self.is_ego = is_ego
        self.lidar_pose = lidar_pose
        self.lidar_range = lidar_range
        self.memory_len = memory_len
        self.require_grad = require_grad
        self.seq_len = seq_len
        self.transformed = True
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = {} # memory FIFO
        self.prepare_data_keys = ['img', 'points', 'annos_global', 'annos_local']

    def update(self, lidar_pose, is_ego, require_grad):
        self.lidar_pose = lidar_pose
        self.is_ego = is_ego
        self.require_grad = require_grad

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(id={self.id}, '
        repr_str += f'is_ego={self.is_ego}, '
        repr_str += f'data={self.data.keys()})'
        return repr_str

    def apply_transform(self):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        DOP.cav_aug_transform(self.data, transform, self.data['augment_params'],
                              apply_to=self.prepare_data_keys)

    def prepare_data(self):
        pass

    def transform_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def has_request(self):
        if 'received_request' in self.data and self.data['received_request'] is not None:
            return True
        else:
            return False

    def get_request_cpm(self):
        return {'lidar_pose': self.lidar_pose}

    def get_response_cpm(self):
        cpm = {}
        for k in ['points']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def receive_request(self, request):
        self.data['received_request'] = request

    def receive_response(self, response):
        self.data['received_response'] = response

    def forward(self, tasks, training_mode, **kwargs):
        self.forward_localization(tasks, training_mode, **kwargs)
        self.forward_local(tasks, training_mode, **kwargs)
        self.forward_fusion(tasks, training_mode, **kwargs)
        self.forward_head(tasks, training_mode, **kwargs)
        return tasks

    def forward_localization(self, tasks, training_mode, **kwargs):
        """To be overloaded."""
        return tasks

    def forward_local(self, tasks, training_mode, **kwargs):
        """To be overloaded."""
        return tasks

    def forward_fusion(self, tasks, training_mode, **kwargs):
        """To be overloaded."""
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        """To be overloaded."""
        return tasks

    def loss(self, tasks, **kwargs):
        """To be overloaded."""
        return tasks

    def reset_data(self, *args, **kwargs):
        del self.data
        self.data = {}

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        pass


class BaseSeqCAV:
    def __init__(self, id, mapped_id, is_ego, lidar_range, memory_len,
                 lidar_pose=None, require_grad=False, seq_len=1, **kwargs):
        self.id = id
        self.mapped_id = mapped_id
        self.is_ego = is_ego
        self.lidar_pose = lidar_pose
        self.lidar_range = lidar_range
        self.memory_len = memory_len
        self.require_grad = require_grad
        self.seq_len = seq_len
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.data = {} # memory FIFO
        self.memory = {}
        self.prepare_data_keys = ['img', 'points', 'annos_global', 'annos_local']

    def update(self, lidar_pose):
        self.lidar_pose = lidar_pose

    def task_id(self, seq_idx):
        return f"{self.id}.{seq_idx}"

    def get_data(self, keys, seq_idx=None):
        if seq_idx is None:
            out = {}
            for i, d in self.data.items():
                out[i] = {}
                for k in keys:
                    out[i][k] = d[k]
        else:
            out = {k: self.data[seq_idx][k] for k in keys}
        return out

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(id={self.id}, '
        repr_str += f'is_ego={self.is_ego}, '
        repr_str += f'data={self.data.keys()})'
        return repr_str

    def apply_transform(self, seq_idx):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        DOP.cav_aug_transform(self.data, transform, self.data['augment_params'],
                              apply_to=self.prepare_data_keys)

    def prepare_data(self, seq_idx):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def has_request(self):
        has_req = False
        for d in self.data.values():
            if 'received_request' in d and d['received_request'] is not None:
                has_req = True
                break
        return has_req

    def get_request_cpm(self):
        return self.get_data(['lidar_poses'])

    def get_response_cpm(self):
        cpm = {}
        for k in ['points']:
            if k in self.data[0]:
                cpm[k] = {i: d[k] for i, d in self.data.items()}
        return cpm

    def receive_request(self, request):
        for i, req in request.items():
            if i not in self.data:
                continue
            self.data[i]['received_request'] = req

    def receive_response(self, response, seq_idx):
        for cav_id, resp in response.items():
            self.data[seq_idx]['received_response'][cav_id] = {k: v[seq_idx] for k, v in resp.items()}

    def forward(self,  tasks, training_mode, seq_idx, with_loss):
        self.prepare_data(seq_idx)
        self.forward_local(tasks, training_mode, seq_idx, with_loss)
        self.forward_fusion(tasks, training_mode, seq_idx, with_loss)
        self.forward_head(tasks, training_mode, seq_idx, with_loss)
        return tasks

    def forward_local(self,  tasks, training_mode, seq_idx, with_loss):
        """To be overloaded."""
        return tasks

    def forward_fusion(self,  tasks, training_mode, seq_idx, with_loss):
        """To be overloaded."""
        return tasks

    def forward_head(self,  tasks, training_mode, seq_idx, with_loss):
        """To be overloaded."""
        return tasks

    def loss(self,  tasks, training_mode, seq_idx, with_loss):
        """To be overloaded."""
        return tasks

    def reset_data(self, *args, **kwargs):
        del self.data
        self.data = {}

    def pre_update_memory(self, seq_idx, **kwargs):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self, seq_idx, **kwargs):
        """Update memory after each forward run of a single frame."""
        pass


class OPV2VtCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']


class OPV2VtCAV_v2(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_ego:
            self.prepare_data_keys = ['points', 'annos_local', 'annos_global', 'annos_global_pred']
        else:
            self.prepare_data_keys = ['points', 'annos_local', 'annos_global']


class DairV2XCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global', 'annos_local']
