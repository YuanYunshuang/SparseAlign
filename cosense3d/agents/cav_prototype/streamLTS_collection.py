import copy

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
            if self.is_ego:
                DOP.apply_transform(self.data, T_c2aug, apply_to=self.prepare_data_keys)
            else:
                data_keys = [k for k in self.prepare_data_keys if k != 'annos_global']
                DOP.apply_transform(self.data, T_c2aug, apply_to=data_keys)
                # global bboxes share the same memory with the ego cav, therefore it is already transformed to the aug coor
                # DOP.apply_transform(self.data, T_e2aug, apply_to=['annos_global'])
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
            DOP.apply_transform(self.data, T_c2aug, apply_to=['points', 'annos_local'])
            self.T_aug2g = T_c2aug

    def prepare_data(self):
        self.prepare_time_scale()
        DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)

    def transform_data(self):
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
        feat = self.data['temp_fusion_feat']
        scores = self.data['detection_local']['all_cls_scores'][-1][...,
                 min(self.data['detection_local']['all_cls_scores'
                     ][-1].shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]
        mask = scores > self.share_score_thr
        cpm['temp_fusion_feat'] = {'ref_pts': feat['ref_pts'][mask], 'outs_dec': feat['outs_dec'][:, mask]}
        return cpm

    def forward_local(self, tasks, training_mode, **kwargs):
        if self.is_ego and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '13:roi_head', {}))

        if self.require_grad and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '14:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '15:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego:
            tasks[grad_mode].append((self.id, '21:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego:
            tasks[grad_mode].append((self.id, '23:det2_head', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.is_ego:
            tasks['loss'].append((self.id, '31:roi_head', {}))
            tasks['loss'].append((self.id, '32:det1_head', {}))
            tasks['loss'].append((self.id, '33:det2_head', {}))
        elif self.require_grad:
            tasks['loss'].append((self.id, '32:det1_head', {}))
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

    def vis_local_detection(self):
        import matplotlib.pyplot as plt
        from cosense3d.utils.vislib import draw_points_boxes_plt
        points = self.data['points'][:, :3].detach().cpu().numpy()
        # pred_boxes = self.data['det_local']['preds']['box'].detach().cpu().numpy()
        gt_boxes = self.data['local_bboxes_3d'][:, :7].detach().cpu().numpy()
        ax = draw_points_boxes_plt(
            pc_range=self.lidar_range.tolist(),
            boxes_gt=gt_boxes[:, :7],
            # boxes_pred=pred_boxes,
            points=points,
            return_ax=True
        )

        ax.set_title('ego' if self.is_ego else 'coop')
        plt.savefig("/home/yuan/Pictures/local_det.png")
        plt.close()

    def vis_local_pred(self):
        import matplotlib.pyplot as plt
        from cosense3d.utils.vislib import draw_points_boxes_plt
        points = self.data['points'][:, :3].detach().cpu().numpy()
        # pred_boxes = self.data['detection_local']['preds']['box'].detach().cpu().numpy()
        ref_pts = self.data['temp_fusion_feat']['ref_pts'].cpu() * (self.lidar_range[3:] - self.lidar_range[:3]) + self.lidar_range[:3]
        ref_pts = ref_pts.detach().numpy()
        gt_boxes = self.data['global_bboxes_3d'][:, :7].detach().cpu().numpy()
        ax = draw_points_boxes_plt(
            pc_range=self.lidar_range.tolist(),
            boxes_gt=gt_boxes[:, :7],
            # boxes_pred=pred_boxes,
            points=points,
            return_ax=True
        )
        ax.plot(ref_pts[:, 0], ref_pts[:, 1], '.r', markersize=1)

        ax.set_title('ego' if self.is_ego else 'coop')
        plt.savefig("/home/yuan/Pictures/local_pred.png")
        plt.close()


class slcDenseToSparse(StreamLidarCAV):

    def prepare_data(self):
        self.prepare_time_scale()

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:roi_head', {}))
        tasks[grad_mode].append((self.id, '13:formatting', {}))
        tasks[grad_mode].append((self.id, '14:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '15:det1_head', {}))


slcFcooper = slcDenseToSparse
slcAttnFusion = slcDenseToSparse


class slcFPVRCNN(StreamLidarCAV):
    def prepare_data(self):
        self.prepare_time_scale()

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:roi_head', {}))
        tasks[grad_mode].append((self.id, '13:keypoint_composer', {}))
        tasks[grad_mode].append((self.id, '14:formatting', {}))
        tasks[grad_mode].append((self.id, '15:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '16:det1_head', {}))

    # def forward_fusion(self, tasks, training_mode, **kwargs):
    #     # if self.is_ego:
    #     #     tasks['with_grad'].append((self.id, '21:spatial_fusion', {}))
    #     return tasks
    #
    # def forward_head(self, tasks, training_mode, **kwargs):
    #     # if self.is_ego:
    #     #     tasks['with_grad'].append((self.id, '23:det2_head', {}))
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
    # def loss(self, tasks, **kwargs):
    #     if self.is_ego:
    #         tasks['loss'].append((self.id, '31:roi_head', {}))
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

    def update_memory_timestamps(self, ref_pts):
        timestamp = torch.zeros_like(ref_pts[..., :1])
        return timestamp


class slcCIASSD(StreamLidarCAV):
    def prepare_data(self):
        self.prepare_time_scale()

    def forward_local(self, tasks, training_mode, **kwargs):
        if (self.is_ego or self.require_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:roi_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        return tasks

    def pre_update_memory(self):
        pass

    def post_update_memory(self):
        pass

    def get_response_cpm(self):
        return {}

    def loss(self, tasks, **kwargs):
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


class LTSDairV2X(StreamLidarCAV):
    def forward_local(self, tasks, training_mode, **kwargs):
        if self.require_grad and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '11:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '12:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '13:roi_head', {}))
        tasks[grad_mode].append((self.id, '14:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '15:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '21:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '23:det2_head', {}))
        return tasks

    def loss(self, tasks, **kwargs):
        if self.require_grad:
            tasks['loss'].append((self.id, '31:roi_head', {}))
            tasks['loss'].append((self.id, '32:det1_head', {}))
        if self.is_ego:
            tasks['loss'].append((self.id, '33:det2_head', {}))
        return tasks


class slcNoBoxTimeDairV2X(LTSDairV2X):

    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)

    def update_memory_timestamps(self, ref_pts):
        timestamp = torch.zeros_like(ref_pts[..., :1])
        return timestamp


class LTSCAVLocCorr(StreamLidarCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.rl_range = torch.nn.Parameter(torch.Tensor([-50, -50, -3.0, 50, 50, 1.0]))
        self.seq_idx = 0

    def apply_transform(self):
        super().apply_transform()
        self.data['lidar_pose_aug'] = self.T_aug2g

    def prepare_data(self):
        self.prepare_time_scale()
        DOP.adaptive_free_space_augmentation(self.data, time_idx=-1)
        # DOP.adaptive_free_space_augmentation(self.data, res=0.5, min_h=0)
        DOP.generate_sparse_target_roadline_points(self.data, range=75)
        self.data['points_rl'] = copy.deepcopy(self.data['points'])
        DOP.filter_range(self.data, self.rl_range, apply_to=['points_rl'])

        # import matplotlib.pyplot as plt
        # points = torch.cat([self.data['points'][:, :3],
        #                     torch.ones_like(self.data['points'][:, :1])], dim=-1)
        # points = (self.data['lidar_poses_gt'] @ points.T)[:3].T.detach().cpu().numpy()
        # rl = self.data['roadline'].detach().cpu().numpy()
        # fig = plt.figure(figsize=(14, 6))
        # ax = fig.add_subplot()
        # ax.plot(points[:, 0], points[:, 1], 'g.', markersize=1)
        # ax.plot(rl[:, 0], rl[:, 1], 'k.', markersize=1)
        # plt.savefig("/home/yys/Downloads/tmp.jpg")
        # plt.close()

    def transform_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def forward_localization(self, tasks, training_mode, **kwargs):
        self.seq_idx = kwargs['seq_idx']
        if self.is_ego and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        if kwargs['seq_idx'] == self.memory_len - 1:
            # only do localization correction for the last frame for easier matching during data fusion
            # the relative transformations between the subsequent frame in the sequence is assumed to be correct.
            tasks[grad_mode].append((self.id, '01:rl_backbone', {}))
            tasks[grad_mode].append((self.id, '02:rl_neck', {}))
            tasks[grad_mode].append((self.id, '03:rlseg_head', {}))
            tasks[grad_mode].append((self.id, '04:localization', {}))

    def forward_local(self, tasks, training_mode, **kwargs):
        if self.is_ego and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '14:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '15:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '16:roi_head', {}))

        if self.require_grad and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '17:temporal_fusion', {}))
        tasks[grad_mode].append((self.id, '18:det1_head', {}))

    def forward_fusion(self, tasks, training_mode, **kwargs):
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego and self.seq_idx == self.memory_len - 1:
            tasks[grad_mode].append((self.id, '21:spatial_fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode, **kwargs):
        grad_mode = 'with_grad' if training_mode else 'no_grad'
        if self.is_ego and self.seq_idx == self.memory_len - 1:
            tasks[grad_mode].append((self.id, '23:det2_head', {}))
        return tasks

    def get_response_cpm(self):
        if self.seq_idx < self.memory_len - 1:
            return {}
        pose_corrected = self.data['lidar_poses_gt']
        pose = self.data['lidar_poses']
        ego_pose = self.data['received_request']['lidar_pose']
        box_ctrs = copy.deepcopy(self.data['detection_local']['preds']['box'][:, :4])
        box_ctrs[:, 3] = 1
        ref_pts = self.data['temp_fusion_feat']['ref_pts']
        lr = self.lidar_range.to(ref_pts.device)
        ref_pts = ref_pts * (lr[3:6] - lr[:3]) + lr[:3]
        # ref_pts = torch.cat([ref_pts, torch.ones_like(ref_pts[:, :1])], dim=-1)
        # transformation matrix from augment-frame to corrected world-frame
        transform = pose_corrected @ pose.inverse() @ self.T_aug2g
        # transform = pose.inverse() @ ego_pose
        box_ctrs = (transform @ box_ctrs.T)[:2].T
        # ref_pts = (transform @ ref_pts.T)[:3].T
        # transform roadline points to corrected world-frame
        roadline = self.data.get('roadline_pred', None)
        roadline = torch.cat([roadline, torch.zeros_like(roadline[:, :1]),
                              torch.ones_like(roadline[:, :1])], dim=-1)
        roadline = (pose_corrected @ roadline.T)[:2].T

        # points is only for GL-visualization
        # points = torch.cat([self.data['points'][:, :3],
        #                     torch.ones_like(self.data['points'][:, :1])], dim=-1)
        # self.data['points'][:, :3] = (transform @ points.T)[:3].T

        # import matplotlib.pyplot as plt
        #
        # pts = self.data['points_rl'].detach().cpu().numpy()
        # rl_vis = self.data.get('roadline_pred', None).detach().cpu().numpy()
        # plt.plot(pts[:, 0], pts[:, 1], 'k.', markersize=1)
        # plt.plot(rl_vis[:, 0], rl_vis[:, 1], 'r.', markersize=1)
        # plt.show()
        # plt.close()

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(6, 6))
        # ax = fig.add_subplot()
        # rl_gt = self.data['points'].detach().cpu().numpy()
        # rl_vis = roadline.detach().cpu().numpy()
        # box_ctrs_vis = box_ctrs.detach().cpu().numpy()
        # ref_pts_vis = ref_pts.detach().cpu().numpy()
        # ax.plot(rl_gt[:, 0], rl_gt[:, 1], 'g.', markersize=1)
        # ax.plot(rl_vis[:, 0], rl_vis[:, 1], 'k.', markersize=1)
        # ax.plot(box_ctrs_vis[:, 0], box_ctrs_vis[:, 1], 'bo', markersize=3)
        # ax.plot(ref_pts_vis[:, 0], ref_pts_vis[:, 1], 'r.', markersize=1)
        # plt.savefig("/home/yys/Downloads/tmp.jpg")
        # plt.close()

        return {
            # 'pose_corrected': self.data['lidar_poses_corrected'],
            'box_ctrs': box_ctrs,
            'roadline': roadline,
            'ref_pts': ref_pts,
            'feat': self.data['temp_fusion_feat']['outs_dec'],
            'Taug2caw': transform,
            'points': self.data['points'],
        }











