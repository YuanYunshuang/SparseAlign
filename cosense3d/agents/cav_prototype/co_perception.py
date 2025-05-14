import torch
import torch_scatter
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseCAV


class StreamLidarCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = kwargs.get('dataset', None)
        self.lidar_range = torch.nn.Parameter(self.lidar_range)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.data['memory'] = None
        self.aug_transform = None
        self.T_aug2g = None
        self.T_g2aug = None
        self.T_e2g = None
        self.use_aug = True

    def refresh_memory(self, prev_exists):
        x = prev_exists.float()
        init_pose = torch.eye(4, device=self.lidar_pose.device).unsqueeze(0).unsqueeze(0)
        if not x:
            self.data['memory'] = {
                'embeddings': x.new_zeros(self.memory_len, self.memory_num_propagated, self.memory_emb_dims),
                'ref_pts': x.new_zeros(self.memory_len, self.memory_num_propagated, self.ref_pts_dim),
                'timestamp': x.new_zeros(self.memory_len, self.memory_num_propagated, 1),
                'pose': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'pose_no_aug': x.new_zeros(self.memory_len, self.memory_num_propagated, 4, 4) ,
                'velo': x.new_zeros(self.memory_len, self.memory_num_propagated, 2),
            }
            self.data['memory']['pose_no_aug'] = self.data['memory']['pose'] + init_pose
            self.aug_transform = None
            self.T_aug2g = None
            self.T_g2aug = None
            self.T_e2g = None
        else:
            for k, v in self.data['memory'].items():
                self.data['memory'][k] = self.data['memory'][k][:self.memory_len] * x
            if not x:
                self.data['memory']['pose_no_aug'][0] = init_pose[0].repeat(self.memory_num_propagated, 1, 1)
        self.data['memory']['prev_exists'] = x

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
            if self.data['prev_exists']:
                self.data['memory']['pose_no_aug'] = T_g2e @ self.data['memory']['pose_no_aug']
                self.data['memory']['ref_pts'] = self.transform_ref_pts(
                    self.data['memory']['ref_pts'], T_g2aug)
            self.data['memory']['pose'] = self.aug_transform @ self.data['memory']['pose_no_aug']


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
        self.prepare_time_scale()
        DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)
        # self.vis_data('transformed', 4)

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

    def update_memory_timestamps(self, ref_pts):
        # transform ref pts to coop coordinates
        transform = self.lidar_pose.inverse() @ self.T_aug2g
        pts = self.transform_ref_pts(ref_pts, transform)
        timestamp = torch.rad2deg(torch.arctan2(pts[:, 1], pts[:, 0])) + 180
        timestamp = - self.data['time_scale'][(timestamp % 360).floor().long()].unsqueeze(-1)
        return timestamp

    def get_response_cpm(self):
        cpm = {}
        for k in ['temp_fusion_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '03:roi_head', {}))
        tasks[grad_mode].append((self.id, '04:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '05:det1_head', {}))

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '11:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '13:det2_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '21:roi_head', {}))
            tasks['loss'].append((self.id, '22:det1_head', {}))
            tasks['loss'].append((self.id, '23:det2_head', {}))
        return tasks

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        if self.data['memory'] is not None:
            self.data['memory']['timestamp'] += self.timestamp
            # pose_inv = self.lidar_pose.inverse()
            # self.data['memory']['pose'] = pose_inv @ self.data['memory']['pose']
            # self.data['memory']['ref_pts'] = self.transform_ref_pts(
            #     self.data['memory']['ref_pts'], pose_inv)

        self.refresh_memory(self.data['prev_exists'])

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        x = self.data['detection_local']
        scores = x['all_cls_scores'][-1][...,
                 min(x['all_cls_scores'][-1].shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]
        topk = torch.topk(scores, k=self.memory_num_propagated).indices

        ref_pts = x['all_bbox_preds'][-1][:, :self.ref_pts_dim]
        velo = x['all_bbox_preds'][-1][:, -2:]
        embeddings = self.data['temp_fusion_feat']['outs_dec'][-1]

        timestamp = self.update_memory_timestamps(ref_pts)
        pose_no_aug = torch.eye(4, device=ref_pts.device).unsqueeze(0).repeat(
            timestamp.shape[0], 1, 1)

        vars = locals()
        for k, v in self.data['memory'].items():
            if k == 'prev_exists' or k == 'pose':
                continue
            rec_topk = vars[k][topk].unsqueeze(0)
            self.data['memory'][k] = torch.cat([rec_topk, v], dim=0)

        # self.vis_ref_pts('post update')

        # ego aug to global
        self.data['memory']['ref_pts'] = self.transform_ref_pts(
            self.data['memory']['ref_pts'], self.T_aug2g)
        self.data['memory']['timestamp'][1:] -= self.timestamp
        self.data['memory']['pose_no_aug'] = self.T_e2g[(None,) * 2] @ self.data['memory']['pose_no_aug'] # aug -->global

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
        else:
            raise NotImplementedError
        return timestamp

    def vis_ref_pts(self, ax=None, label=None, his_len=1, **kwargs):
        import matplotlib.pyplot as plt
        from cosense3d.utils.vislib import draw_points_boxes_plt
        if ax is None:
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot()
        pcd = self.data['points'][:, :3].detach().cpu().numpy()
        gt_boxes = self.data['local_bboxes_3d'].detach().cpu().numpy()
        ax = draw_points_boxes_plt(
            pc_range=self.lidar_range.tolist(),
            boxes_gt=gt_boxes[:, :7],
            points=pcd,
            ax=ax,
            return_ax=True,
        )

        ref_pts = self.data['memory']['ref_pts'].detach().cpu().numpy()
        markers = ['.r', '.m', '.b', '.c']
        for i in range(his_len):
            plt.plot(ref_pts[i, :, 0], ref_pts[i, :, 1], markers[i], markersize=2)
        ax.set_title(f"{label}: {self.data['scenario']}, {self.data['frame']}")
        plt.show()
        plt.close()

        return ax

    def vis_poses(self, ax=None, label=None, his_len=1, **kwargs):
        import matplotlib.pyplot as plt
        markers = ['r', 'm', 'b', 'c']
        mem_poses = self.data['memory']['pose'][:, 0].detach().cpu()
        p0 = mem_poses[:his_len, :2, -1].numpy()
        p1 = mem_poses[:his_len] @ torch.tensor([1., 0., 0., 1.]).view(1, 4, 1).repeat(his_len, 1, 1)
        p2 = mem_poses[:his_len] @ torch.tensor([0., 1., 0., 1.]).view(1, 4, 1).repeat(his_len, 1, 1)
        p1 = p1.squeeze(-1)[:, :2].numpy()
        p2 = p2.squeeze(-1)[:, :2].numpy()

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.axis('equal')
        for i in range(his_len):
            ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]], markers[i])
            ax.plot([p0[i, 0], p2[i, 0]], [p0[i, 1], p2[i, 1]], markers[i])
        return ax


class slcDenseToSparse(StreamLidarCAV):

    def prepare_data(self):
        self.prepare_time_scale()
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:roi_head', {}))
        tasks[grad_mode].append((self.id, '03:formatting', {}))
        tasks[grad_mode].append((self.id, '04:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '05:det1_head', {}))


slcFcooper = slcDenseToSparse
slcAttnFusion = slcDenseToSparse


class slcFPVRCNN(StreamLidarCAV):
    def prepare_data(self):
        self.prepare_time_scale()
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:roi_head', {}))
        tasks[grad_mode].append((self.id, '03:keypoint_composer', {}))
        tasks[grad_mode].append((self.id, '04:formatting', {}))
        tasks[grad_mode].append((self.id, '05:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '06:det1_head', {}))

    # def forward_fusion(self, tasks, training_mode):
    #     # if self.is_ego:
    #     #     tasks['with_grad'].append((self.id, '11:spatial_fusion', {}))
    #     return tasks
    #
    # def forward_head(self, tasks, training_mode):
    #     # if self.is_ego:
    #     #     tasks['with_grad'].append((self.id, '13:det2_head', {}))
    #     return tasks
    #
    # def pre_update_memory(self):
    #     pass
    #
    # def post_update_memory(self):
    #     pass
    #
    # def get_response_cpm(self):
    #     return {}
    #
    # def loss(self, tasks):
    #     if self.is_ego:
    #         tasks['loss'].append((self.id, '21:roi_head', {}))
    #     return tasks
    #
    # def apply_transform(self):
    #     if self.use_aug:
    #         if self.is_ego:
    #             T_e2g = self.lidar_pose
    #             T_g2e = self.lidar_pose.inverse()
    #             T_c2e = torch.eye(4).to(self.lidar_pose.device)
    #         else:
    #             # cav to ego
    #             T_e2g = self.data['received_request']['lidar_pose']
    #             T_g2e = self.data['received_request']['lidar_pose'].inverse()
    #             T_c2e = T_g2e @ self.lidar_pose
    #
    #         if self.aug_transform is None:
    #             self.aug_transform = DOP.update_transform_with_aug(
    #                 torch.eye(4).to(self.lidar_pose.device), self.data['augment_params'])
    #             T_e2aug = self.aug_transform
    #         else:
    #             # adapt aug params to the current ego frame
    #             T_e2aug = self.T_g2aug @ T_e2g
    #
    #         T_c2aug = T_e2aug @ T_c2e
    #         T_g2aug = T_e2aug @ T_g2e
    #
    #         DOP.apply_transform(self.data, T_c2aug, apply_to=self.prepare_data_keys)
    #
    #         self.T_e2g = T_e2g
    #         self.T_g2aug = T_g2aug
    #         self.T_aug2g = T_g2aug.inverse() # ego aug to global
    #
    #     else:
    #         if self.is_ego:
    #             transform = torch.eye(4).to(self.lidar_pose.device)
    #         else:
    #             # cav to ego
    #             request = self.data['received_request']
    #             transform = request['lidar_pose'].inverse() @ self.lidar_pose
    #
    #         T_c2aug = DOP.update_transform_with_aug(transform, self.data['augment_params'])
    #         DOP.apply_transform(self.data, T_c2aug, apply_to=self.prepare_data_keys)
    #         self.T_aug2g = T_c2aug


class slcNoBoxTime(StreamLidarCAV):

    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def update_memory_timestamps(self, ref_pts):
        timestamp = torch.zeros_like(ref_pts[..., :1])
        return timestamp


class slcCIASSD(StreamLidarCAV):
    def prepare_data(self):
        self.prepare_time_scale()
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:roi_head', {}))

    def forward_fusion(self, tasks, training_mode):
        return tasks

    def forward_head(self, tasks, training_mode):
        return tasks

    def pre_update_memory(self):
        pass

    def post_update_memory(self):
        pass

    def get_response_cpm(self):
        return {}

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '21:roi_head', {}))
        return tasks

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


class StreamLidarCAVLocCorr(StreamLidarCAV):
    def get_response_cpm(self):
        cpm = {}
        cpm['coop_det_ctr'] = self.data['detection_local']['preds']['box'][:, :3]
        for k in ['temp_fusion_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '11:spatial_alignment', {}))
            tasks['with_grad'].append((self.id, '12:spatial_fusion', {}))
        return tasks






