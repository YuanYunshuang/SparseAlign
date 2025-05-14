import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseCAV


class RLsegCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = kwargs.get('dataset', None)
        self.lidar_range = torch.nn.Parameter(self.lidar_range)
        self.prepare_data_keys = ['points', 'annos_local', 'roadline_tgts']
        self.aug_transform = None
        self.T_aug2g = None
        self.T_g2aug = None
        self.T_e2g = None
        self.use_aug = True

    def apply_transform(self):
        if self.use_aug:
            if self.is_ego:
                T_e2g = self.lidar_pose
                T_g2e = self.lidar_pose.inverse()
                T_c2e = torch.eye(4).to(self.lidar_pose.device)
            else:
                # cav to ego
                T_e2g = self.data['received_request']['lidar_pose']
                T_g2e = self.data['received_request']['lidar_pose'].inverse()
                T_c2e = T_g2e @ self.lidar_pose

            if self.aug_transform is None:
                self.aug_transform = DOP.update_transform_with_aug(
                    torch.eye(4).to(self.lidar_pose.device), self.data['augment_params'])
                T_e2aug = self.aug_transform
            else:
                # adapt aug params to the current ego frame
                T_e2aug = self.T_g2aug @ T_e2g

            T_c2aug = T_e2aug @ T_c2e
            T_g2aug = T_e2aug @ T_g2e

            DOP.apply_transform(self.data, T_c2aug, apply_to=self.prepare_data_keys)

            self.T_e2g = T_e2g
            self.T_g2aug = T_g2aug
            self.T_aug2g = T_g2aug.inverse() # ego aug to global

        else:
            if self.is_ego:
                transform = torch.eye(4).to(self.lidar_pose.device)
            else:
                # cav to ego
                request = self.data['received_request']
                transform = request['lidar_pose'].inverse() @ self.lidar_pose

            T_c2aug = DOP.update_transform_with_aug(transform, self.data['augment_params'])
            DOP.apply_transform(self.data, T_c2aug, apply_to=self.prepare_data_keys)
            self.T_aug2g = T_c2aug

    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data, res=0.5, min_h=0)
        DOP.generate_sparse_target_roadline_points(self.data)

    def get_request_cpm(self):
        return {'lidar_pose': self.lidar_pose}

    def get_response_cpm(self):
        cpm = {}
        for k in ['bev_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:backbone_neck', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '21:rlseg_head', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.is_ego:
            tasks['loss'].append((self.id, '31:rlseg_head', {}))
        return tasks



