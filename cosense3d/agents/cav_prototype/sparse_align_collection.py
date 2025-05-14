import torch
import torch_scatter

from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseCAV
from cosense3d.utils.box_utils import transform_boxes_3d


class StreamLidarAlignCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = kwargs.get('dataset', None)
        self.lidar_range = torch.nn.Parameter(self.lidar_range)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.data['memory'] = None
        self.aug_transform = None
        self.T_aug2g = None
        self.T_g2aug = None
        self.T_l2g = None
        self.T_l2aug = None
        self.transformed = False
        self.alignment = 'ego'
        self.FSA = kwargs.get('FSA', False)

    def refresh_memory(self, prev_exists):
        x = prev_exists.float()
        init_pose = torch.eye(4, device=self.lidar_pose.device).unsqueeze(0).unsqueeze(0)
        if not x:
            self.data['memory'] = {
                'embeddings': x.new_zeros(self.memory_len, self.memory_num_propagated, self.memory_emb_dims),
                'ref_pts': x.new_zeros(self.memory_len, self.memory_num_propagated, self.ref_pts_dim),
                'timestamp': x.new_zeros(self.memory_len, self.memory_num_propagated, 1),
                'pose': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                # 'pose_no_aug': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'velo': x.new_zeros(self.memory_len, self.memory_num_propagated, 2),
            }
            self.data['memory']['pose'] = self.data['memory']['pose'] + init_pose
            self.aug_transform = None
            self.T_aug2g = None
            self.T_g2aug = None
            self.T_l2g = None
        else:
            for k, v in self.data['memory'].items():
                self.data['memory'][k] = self.data['memory'][k][:self.memory_len] * x
            if not x:
                self.data['memory']['pose'][0] = init_pose[0].repeat(self.memory_num_propagated, 1, 1)
        self.data['memory']['prev_exists'] = x

    def prepare_data(self):
        self.prepare_time_scale()
        if self.FSA:
            DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)
        self.prepare_transform()

    def prepare_transform(self):
        T_l2g = self.data['lidar_poses']
        T_g2l = T_l2g.inverse()

        if self.aug_transform is None:
            self.aug_transform = DOP.update_transform_with_aug(
                torch.eye(4).to(self.lidar_pose.device), self.data['augment_params'])
            # self.aug_transform = torch.eye(4).to(self.lidar_pose.device)  # no aug
            T_l2aug = self.aug_transform
        else:
            # adapt aug params to the current ego frame
            T_l2aug = self.T_g2aug @ T_l2g

        self.T_l2aug = T_l2aug
        self.T_l2g = T_l2g
        self.T_g2aug = T_l2aug @ T_g2l
        self.T_aug2g = self.T_g2aug.inverse() # aug to global
        self.data['T_aug2g'] = self.T_aug2g

        if not self.is_ego and self.data['global_bboxes_3d'] is not None:
            self.data['global_bboxes_3d'] = self.data['global_bboxes_3d'].clone().detach()

    def transform_data(self):
        self.apply_transform()
        if self.is_ego:
            DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)
        else:
            # ego_pose = self.data['received_request']['T_aug2g']
            # Tc2e = ego_pose.inverse() @ self.T_aug2g
            DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys,
                             # transform=Tc2e
                             )
            # self.vis_tacood_scene()

    def prepare_time_scale(self):
        # hash time
        azi = torch.arctan2(self.data['points'][:, 1], self.data['points'][:, 0])
        azi, inds = (torch.rad2deg(azi) + 180).floor().long().unique(return_inverse=True)
        times = torch.zeros_like(azi).float()
        torch_scatter.scatter_mean(self.data['points'][:, -1], inds, dim=0, out=times)
        if len(times) < 360:
            time360 = times.new_zeros(360)
            time360[azi] = times
            time360[time360 == 0] = times.mean()
        else:
            time360 = times
        self.data['time_scale'] = time360
        self.data['time_scale_reduced'] = time360 - self.timestamp
        # self.data['points'] = self.data['points'][:, :-1]

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        # tasks[grad_mode].append((self.id, '12:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '13:roi_head', {}))

        if self.require_grad and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '14:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '15:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if kwargs['seq_idx'] < self.seq_len - 1:
            return tasks
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego:
            tasks[grad_mode].append((self.id, '21:spatial_alignment', {}))
            tasks[grad_mode].append((self.id, '22:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if kwargs['seq_idx'] < self.seq_len - 1:
            return tasks
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego:
            tasks[grad_mode].append((self.id, '23:det2_head', {}))
        return tasks

    def get_response_cpm(self):
        feat = self.data['temp_fusion_feat']
        scores = self.data['detection_local']['all_cls_scores'][-1]
        scores = scores[..., min(scores.shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]
        mask = scores > self.share_score_thr
        cpm = {'ref_pts': feat['ref_pts'][mask],
               'outs_dec': feat['outs_dec'][mask],
               'pose': self.data['T_aug2g'],
               'preds': self.data['detection_local'].get('preds', None)}
        return cpm

    def apply_transform(self):
        if self.is_ego:
            DOP.apply_transform(self.data, self.T_l2aug, apply_to=self.prepare_data_keys)
        else:
            # data_keys = [k for k in self.prepare_data_keys if k != 'annos_global']
            # DOP.apply_transform(self.data, self.T_l2aug, apply_to=data_keys)
            if self.data['global_bboxes_3d'] is not None and self.require_grad:
                ego_pose = self.data['received_request']['lidar_poses_gt']
                ego_to_coop = self.data['lidar_poses_gt'].inverse() @ ego_pose
                self.data['global_bboxes_3d'][:, :7] = transform_boxes_3d(self.data['global_bboxes_3d'][:, :7], ego_to_coop)
            DOP.apply_transform(self.data, self.T_l2aug, apply_to=self.prepare_data_keys)
            # global bboxes share the same memory with the ego cav, therefore it is already transformed to the aug coor
            # DOP.apply_transform(self.data, T_e2aug, apply_to=['annos_global'])
        if self.data['prev_exists']:
            self.data['memory']['pose'] = self.T_aug2g.inverse() @ self.data['memory']['pose'] # global -> aug
            self.data['memory']['ref_pts'] = self.transform_ref_pts(
                self.data['memory']['ref_pts'], self.T_g2aug)

    def get_request_cpm(self):
        return {'T_aug2g': self.T_aug2g, 'lidar_poses': self.lidar_pose, 'lidar_poses_gt': self.data['lidar_poses_gt'],}

    def loss(self, tasks, **kwargs):
        if self.is_ego or (self.require_grad and kwargs['seq_idx'] == self.seq_len - 1):
            tasks['loss'].append((self.id, '31:roi_head', {}))
        if self.is_ego:
            tasks['loss'].append((self.id, '32:det1_head', {}))
            tasks['loss'].append((self.id, '33:det2_head', {}))
            # tasks['loss'].append((self.id, '34:spatial_alignment', {}))
        elif self.require_grad:
            tasks['loss'].append((self.id, '32:det1_head', {}))
        return tasks

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        if self.data['memory'] is not None:
            self.data['memory']['timestamp'] += self.timestamp

        self.refresh_memory(self.data['prev_exists'])

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        x = self.data['detection_local']
        scores = x['all_cls_scores'][-1][...,
                 min(x['all_cls_scores'][-1].shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]

        if scores.shape[0] < self.memory_num_propagated:
            sort_inds = scores.argsort(descending=True)
            n_repeat = self.memory_num_propagated // len(scores) + 1
            sort_inds = torch.cat([sort_inds] * n_repeat, dim=0)
            topk = sort_inds[:self.memory_num_propagated]
        else:
            topk = torch.topk(scores, k=self.memory_num_propagated).indices

        ref_pts = x['ref_pts'][:, :self.ref_pts_dim]
        velo = x['all_bbox_preds'][-1][:, -2:]
        embeddings = self.data['temp_fusion_feat']['outs_dec'][:, -1]

        timestamp = self.update_memory_timestamps(ref_pts)
        pose = torch.eye(4, device=ref_pts.device).unsqueeze(0).repeat(
            timestamp.shape[0], 1, 1)

        vars = locals()
        for k, v in self.data['memory'].items():
            if k == 'prev_exists' or k == 'pose':
                continue
            rec_topk = vars[k][topk].unsqueeze(0)
            self.data['memory'][k] = torch.cat([rec_topk, v], dim=0)

        # self.vis_ref_pts(label='post update', his_len=4)

        # aug to global
        self.data['memory']['ref_pts'] = self.transform_ref_pts(
            self.data['memory']['ref_pts'], self.T_aug2g)
        self.data['memory']['timestamp'][1:] -= self.timestamp
        self.data['memory']['pose'] = self.T_aug2g[(None,) * 2] @ self.data['memory']['pose'] # aug -->global

        # if self.require_grad:
        #     # self.vis_local_detection()
        #     self.vis_local_pred()
        #     print('d')

    def transform_ref_pts(self, reference_points, matrix):
        reference_points = torch.cat(
            [reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
        if reference_points.ndim == 3:
            reference_points = matrix.unsqueeze(0) @ reference_points.permute(0, 2, 1)
            reference_points = reference_points.permute(0, 2, 1)[..., :3]
        elif reference_points.ndim == 2:
            reference_points = matrix @ reference_points.T
            reference_points = reference_points.T[..., :3]
        else:
            raise NotImplementedError
        return reference_points

    @property
    def timestamp(self):
        if self.dataset == 'opv2vt':
            timestamp = float(self.data['frame']) * 0.1 / 2
        elif self.dataset == 'dairv2xt':
            timestamp = self.data['global_time']
        elif self.dataset == 'opv2v':
            timestamp = self.data['timestamp']
        elif self.dataset == 'dairv2x':
            timestamp = self.data['timestamp'] - 1.62616*1e9
        else:
            raise NotImplementedError
        return timestamp


class slcNoBoxTime(StreamLidarAlignCAV):

    def prepare_data(self):
        if self.FSA:
            DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)
        self.prepare_transform()

    def update_memory_timestamps(self, ref_pts):
        timestamp = torch.zeros_like(ref_pts[..., :1])
        return timestamp


class slcOPV2V(StreamLidarAlignCAV):

    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data)
        self.prepare_transform()

    def update_memory_timestamps(self, ref_pts):
        timestamp = torch.zeros_like(ref_pts[..., :1]) - self.timestamp
        return timestamp


class slcDenseToSparse(StreamLidarAlignCAV):

    def prepare_data(self):
        self.prepare_time_scale()
        self.prepare_transform()

    def forward_local(self, tasks, training_mode, **kwargs):
        if self.require_grad and training_mode or (self.require_grad and kwargs['seq_idx'] == self.seq_len - 1):
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:roi_head', {}))
        tasks[grad_mode].append((self.id, '13:formatting', {}))

        if self.require_grad and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '14:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '15:det1_head', {}))


class slcFPVRCNN(StreamLidarAlignCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_train = False

    def prepare_data(self):
        self.prepare_time_scale()
        self.prepare_transform()

    def forward_local(self, tasks, training_mode, **kwargs):
        if self.require_grad and training_mode or (self.require_grad and kwargs['seq_idx'] == self.seq_len - 1):
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:roi_head', {}))
        if not self.pre_train:
            # tasks[grad_mode].append((self.id, '13:keypoint_composer', {}))
            tasks[grad_mode].append((self.id, '14:formatting', {}))
            if self.require_grad and training_mode:
                grad_mode = 'with_grad'
            else:
                grad_mode = 'no_grad'
            tasks[grad_mode].append((self.id, '15:temporal_fusion', {}))
            tasks[grad_mode].append((self.id, '16:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if not self.pre_train:
            super().forward_fusion(tasks, training_mode, **kwargs)
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if not self.pre_train:
            super().forward_head(tasks, training_mode, **kwargs)
        return tasks

    def pre_update_memory(self):
        if not self.pre_train:
            super().pre_update_memory()

    def post_update_memory(self):
        if not self.pre_train:
            super().post_update_memory()

    def get_response_cpm(self):
        if self.pre_train:
            return {}
        else:
            return super().get_response_cpm()

    def loss(self, tasks, **kwargs):
        if self.pre_train:
            if self.is_ego or 'dairv2x' in self.dataset:
                tasks['loss'].append((self.id, '31:roi_head', {}))
        else:
            super().loss(tasks, **kwargs)
        return tasks

    def apply_transform(self):
        if self.is_ego:
            DOP.apply_transform(self.data, self.T_l2aug, apply_to=self.prepare_data_keys)
        else:
            # data_keys = [k for k in self.prepare_data_keys if k != 'annos_global']
            # DOP.apply_transform(self.data, self.T_l2aug, apply_to=data_keys)
            if self.data['global_bboxes_3d'] is not None and self.require_grad:
                ego_pose = self.data['received_request']['lidar_poses_gt']
                ego_to_coop = self.data['lidar_poses_gt'].inverse() @ ego_pose
                self.data['global_bboxes_3d'][:, :7] = transform_boxes_3d(self.data['global_bboxes_3d'][:, :7], ego_to_coop)
            DOP.apply_transform(self.data, self.T_l2aug, apply_to=self.prepare_data_keys)
            # global bboxes share the same memory with the ego cav, therefore it is already transformed to the aug coor
            # DOP.apply_transform(self.data, T_e2aug, apply_to=['annos_global'])
        if not self.pre_train and self.data['prev_exists']:
            self.data['memory']['pose'] = self.T_aug2g.inverse() @ self.data['memory']['pose'] # global -> aug
            self.data['memory']['ref_pts'] = self.transform_ref_pts(
                self.data['memory']['ref_pts'], self.T_g2aug)


















