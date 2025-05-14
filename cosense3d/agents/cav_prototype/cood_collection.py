from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseCAV


class CoodCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global']

    def get_response_cpm(self):
        cpm = {}
        for k in ['pts_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            tasks['with_grad'].append((self.id, '11:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '11:pts_backbone', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '21:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '22:detection_head', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.is_ego:
            tasks['loss'].append((self.id, '32:detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}


class FpvrcnnCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global', 'annos_local']

    def get_response_cpm(self):
        cpm = {}
        for k in ['keypoint_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            tasks['with_grad'].append((self.id, '11:pts_backbone', {}))
            tasks['with_grad'].append((self.id, '12:detection_head_local', {}))
            tasks['with_grad'].append((self.id, '13:keypoint_composer', {}))
        else:
            tasks['no_grad'].append((self.id, '11:pts_backbone', {}))
            tasks['no_grad'].append((self.id, '12:detection_head_local', {}))
            tasks['no_grad'].append((self.id, '13:keypoint_composer', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '21:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '22:detection_head_global', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.is_ego:
            tasks['loss'].append((self.id, '31:detection_head_local', {}))
            tasks['loss'].append((self.id, '32:detection_head_global', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}


class Sp3DCAV(CoodCAV):
    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data)

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '21:fusion', {}))
            tasks['with_grad'].append((self.id, '22:fusion_neck', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '23:bev_head', {}))
            tasks['with_grad'].append((self.id, '24:detection_head', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.is_ego:
            tasks['loss'].append((self.id, '31:bev_head', {}))
            tasks['loss'].append((self.id, '32:detection_head', {}))
        return tasks



