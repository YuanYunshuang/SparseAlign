import copy
import os
import random

import matplotlib.pyplot as plt
import torch
import torch_scatter

from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.agents.utils.transform import generate_bev_tgt_pts, transform_points
from cosense3d.utils.box_utils import transform_boxes_3d


class DataManager:
    def __init__(self,
                 cav_manager,
                 lidar_range,
                 voxel_size=None,
                 aug=None,
                 pre_process=[],
                 loc_err=None):
        self.cav_manager = cav_manager
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size
        self.aug = aug
        self.pre_process = pre_process
        self.loc_err = loc_err

    def apply_preprocess(self):
        if isinstance(self.pre_process, list):
            for p in self.pre_process:
                getattr(self, p)()
        elif isinstance(self.pre_process, dict):
            for p, args in self.pre_process.items():
                getattr(self, p)(**args)

    def remove_global_empty_boxes(self):
        for i, cavs in enumerate(self.cav_manager.cavs):
            if cavs[0].data.get('global_bboxes_3d', None) is None:
                continue
            assert cavs[0].is_ego
            points = []
            for cav in cavs:
                if cav.is_ego or cav.transformed:
                    points.append(cav.data['points'])
                else:
                    transform = cavs[0].T_g2aug @ cav.T_aug2g
                    pts = transform_points(cav.data['points'].clone(), transform)
                    points.append(pts)
            points_vis = [pts.cpu().numpy().T for pts in points]
            points = torch.cat(points, dim=0)
            points = points[points[:, 3]>=0]
            global_boxes = cavs[0].data['global_bboxes_3d']
            box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3],
                                          global_boxes.unsqueeze(0)[..., :7])[0]
            box_idx = box_idx[box_idx > -1]
            num_pts = torch.zeros_like(global_boxes[:, 0]).long()
            torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
            mask = num_pts > 3
            cavs[0].data['global_bboxes_3d'] = global_boxes[mask]
            cavs[0].data['global_labels_3d'] = cavs[0].data['global_labels_3d'][mask]
            if 'bboxes_3d_pred' in cavs[0].data:
                cavs[0].data['bboxes_3d_pred'] = cavs[0].data['bboxes_3d_pred'][:, mask]

            # if i == 0:
            #     from cosense3d.utils.vislib import draw_points_boxes_plt
            #     ax = draw_points_boxes_plt(
            #         pc_range=100,
            #         boxes_gt=cavs[0].data['global_bboxes_3d'].cpu().numpy(),
            #         return_ax=True,
            #     )
            #     for pts in points_vis:
            #         ax.plot(pts[0], pts[1], '.', markersize=1)
            #     plt.savefig(f"{os.environ['HOME']}/Downloads/tmp1.jpg")
            #     plt.close()

    def update_tacood_bboxes_for_coop(self):
        for cavs in self.cav_manager.cavs:
            if cavs[0].data['global_bboxes_3d'] is None:
                continue
            for cav in cavs[1:]:
                if cav.require_grad and not cav.transformed:
                    try:
                        ego_pose = cavs[0].data['lidar_poses_gt']
                        coop_pose = cav.data['lidar_poses_gt']
                    except:
                        ego_pose = cavs[0].data['lidar_poses']
                        coop_pose = cav.data['lidar_poses']

                    ego_to_coop = coop_pose.inverse() @ ego_pose
                    boxes = cavs[0].data['global_bboxes_3d'].clone()
                    boxes[:, :7] = transform_boxes_3d(boxes[:, :7], ego_to_coop)
                    cav.data['global_bboxes_3d'] = boxes
                    cav.data['global_labels_3d'] = cavs[0].data['global_labels_3d']
                    cav.data['global_bboxes_id'] = cavs[0].data['global_bboxes_id']
                    cav.data['global_names'] = cavs[0].data['global_names']


    def generate_global_non_empty_mask(self):
        for cavs in self.cav_manager.cavs:
            if cavs[0].data.get('global_bboxes_3d', None) is None:
                continue
            assert cavs[0].is_ego
            points = torch.cat([cav.data['points'] for cav in cavs], dim=0)
            global_boxes = cavs[0].data['global_bboxes_3d']
            box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3],
                                          global_boxes.unsqueeze(0)[..., :7])[0]
            box_idx = box_idx[box_idx > -1]
            num_pts = torch.zeros_like(global_boxes[:, 0]).long()
            torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
            mask = num_pts > 3
            cavs[0].data['global_bboxes_mask'] = mask

    def remove_local_empty_boxes(self, ego_only=False):
        for cavs in self.cav_manager.cavs:
            for cav in cavs:
                if not cav.is_ego and ego_only:
                    continue
                if cav.data.get('local_bboxes_3d', None) is None:
                    continue
                points = cav.data['points'][cav.data['points'][:, 3]>=0]
                local_boxes = cav.data['local_bboxes_3d']
                box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3],
                                              local_boxes.unsqueeze(0)[..., :7])[0]
                box_idx = box_idx[box_idx > -1]
                num_pts = torch.zeros_like(local_boxes[:, 0]).long()
                torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
                mask = num_pts > 3
                cav.data['local_bboxes_3d'] = local_boxes[mask]
                cav.data['local_labels_3d'] = cav.data['local_labels_3d'][mask]

    def generate_local_non_empty_mask(self, ego_only=False):
        for cavs in self.cav_manager.cavs:
            for cav in cavs:
                if not cav.is_ego and ego_only:
                    continue
                if cav.data.get('local_bboxes_3d', None) is None:
                    continue
                points = cav.data['points']
                local_boxes = cav.data['local_bboxes_3d']
                box_idx = points_in_boxes_gpu(points.unsqueeze(0)[..., :3],
                                              local_boxes.unsqueeze(0)[..., :7])[0]
                box_idx = box_idx[box_idx > -1]
                num_pts = torch.zeros_like(local_boxes[:, 0]).long()
                torch_scatter.scatter_add(torch.ones_like(box_idx), box_idx, dim=0, out=num_pts)
                mask = num_pts > 3
                cav.data['local_bboxes_mask'] = mask

    def sample_global_bev_tgt_pts(self, sam_res=0.4, map_res=0.2, range=50, max_num_pts=5000, discrete=False):
        for cavs in self.cav_manager.cavs:
            assert cavs[0].is_ego
            points = torch.cat([cav.data['points'] for cav in cavs], dim=0)
            transform = cavs[0].T_e2g.inverse() @ cavs[0].T_aug2g
            bev_pts = generate_bev_tgt_pts(points, cavs[0].data, transform,
                                           sam_res, map_res, range, max_num_pts, discrete)
            cavs[0].data['global_bev_tgt_pts'] = bev_pts

            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            # lidar = points.cpu().numpy()
            # pts = bev_pts.cpu().numpy()
            # pos = pts[:, 2] == 1
            # neg = pts[:, 2] == 0
            #
            # ax = draw_points_boxes_plt(
            #     pc_range=50,
            #     points=pts[pos, :],
            #     points_c='r',
            #     return_ax=True
            # )
            # ax.plot(pts[neg, 0], pts[neg, 1], '.', c='b', markersize=1)
            # ax.plot(lidar[:, 0], lidar[:, 1], '.', c='gray', markersize=1)
            # plt.savefig("/home/yuan/Downloads/tmp.png")
            # plt.close()

    def distribute_to_seq_list(self, batch_dict, seq_len):
        result = []
        for l in range(seq_len):
            res = {}
            for k, v in batch_dict.items():
                x_list = [x[l] for x in v if l < len(x)]
                if len(x_list) == 0:
                    res = {}
                    break
                res[k] = x_list
            if len(res) > 0:
                result.append(res)
        return result

    def distribute_to_cav(self, valid_agent_ids=None, **data):
        cavs = self.cav_manager.cavs
        global_data_list = []
        for b, agent_ids in enumerate(valid_agent_ids):
            global_data = {}
            for j, ai in enumerate(agent_ids):
                assert cavs[b][j].id == f'{b}.{ai}'
                for k, v in data.items():
                    if isinstance(v[b], list) and len(v[b]) == len(agent_ids):
                        cavs[b][j].data[k] = v[b][j]
                    elif k == 'chosen_cams':
                        cavs[b][j].data[k] = v[b][ai]
                    elif k == 'augment_params':
                        cavs[b][j].data[k] = v[b]
                        global_data[k] = v[b]
                    else:
                        cavs[b][j].data[k] = v[b]
            global_data_list.append(global_data)
        return global_data_list

    def distribute_to_seq_cav(self, data):
        for l, d in enumerate(data):
            valid_agent_ids = d['valid_agent_ids']
            global_data_list = []
            for b, agent_ids in enumerate(valid_agent_ids):
                global_data = {}
                for j, ai in enumerate(agent_ids):
                    new_data = {}
                    for k, v in d.items():
                        if isinstance(v[b], list) and len(v[b]) == len(agent_ids):
                            new_data[k] = v[b][j]
                        elif k == 'chosen_cams':
                            new_data[k] = v[b][ai]
                        elif k == 'augment_params':
                            new_data[k] = v[b]
                            global_data[k] = v[b]
                        # elif cavs[b][j].is_ego:
                        else:
                            new_data[k] = v[b]
                    self.cav_manager.get_cav_with_id(f'{b}.{ai}').data[l] = new_data
                global_data_list.append(global_data)
        return global_data_list

    def generate_augment_params(self, batch_dict, seq_len):
        B = len(batch_dict['scenario'])
        if self.aug is None:
            rand_aug = [[None] * seq_len] * B
        else:
            rand_aug = []
            def rand_from_range(r):
                return torch.rand(1) * (r[1] - r[0]) + r[0]
            for i in range(B):
                cur_aug = {}
                if 'rot_range' in self.aug:
                    theta = rand_from_range(self.aug['rot_range'])
                    ct = torch.cos(theta)
                    st = torch.sin(theta)
                    transform = torch.eye(4)
                    transform[0, 0] = ct
                    transform[0, 1] = -st
                    transform[1, 0] = st
                    transform[1, 1] = ct
                    cur_aug['rot'] = transform
                if 'trans_std' in self.aug:
                    cur_aug['trans'] = torch.randn(len(self.aug['trans_std'])) * torch.tensor(self.aug['trans_std'])
                if 'scale_ratio_range' in self.aug:
                    cur_aug['scale'] = rand_from_range(self.aug['scale_ratio_range'])
                if 'flip' in self.aug:
                    cur_aug['flip'] = {'flip_idx': random.randint(0, 3), 'flip_axis': self.aug['flip']}
                rand_aug.append([cur_aug for _ in range(seq_len)])
        batch_dict['augment_params'] = rand_aug

    def add_loc_err(self, batch_dict, seq_len):
        if self.loc_err is None:
            return
        # TODO

    def gather(self, cav_list, data_keys):
        data_list = []
        for k in data_keys:
            data = []
            for cav_id in cav_list:
                try:
                    data.append(self.cav_manager.get_cav_with_id(cav_id).data[k])
                except:
                    print('d')
            data_list.append(data)
        return data_list

    def scatter(self, cav_list, data_dict):
        for k, data_list in data_dict.items():
            for cav_id, data in zip(cav_list, data_list):
                self.update(cav_id, k, data)

    def update(self, cav_id, data_key, data):
        self.cav_manager.get_cav_with_id(cav_id).data[data_key] = data

    def gather_batch(self, batch_idx, key, to_numpy=False):
        data = {}
        ego_cav = self.cav_manager.cavs[batch_idx][0]
        for cav in self.cav_manager.cavs[batch_idx]:
            if key not in cav.data:
                continue
            d = cav.data[key]
            if key == 'points' and not cav.transformed:
                T = ego_cav.T_g2aug @ cav.T_aug2g
                d = transform_points(d.clone(), T)
            elif key == 'local_bboxes_3d' and not cav.transformed:
                T = ego_cav.T_g2aug @ cav.T_aug2g
                d = d.clone()
                d[:, :7] = transform_boxes_3d(d[:, :7], T)
            elif key == 'detection_local' and not cav.transformed:
                T = ego_cav.T_g2aug @ cav.T_aug2g
                try:
                    d = copy.deepcopy(d)
                except:
                    print("deep copy failed, modifying original data for visualization. ")
                d['preds']['box'][:, :7] = transform_boxes_3d(d['preds']['box'][:, :7], T)
            if isinstance(d, torch.Tensor) and to_numpy:
                d = d.cpu().numpy()
            elif isinstance(d, list) and len(d) > 0 and isinstance(d[0], torch.Tensor):
                d = [x.cpu().numpy() for x in d]
            data[cav.id] = d
        return data

    def gather_ego_data(self, key):
        data = {}
        for cavs in self.cav_manager.cavs:
            assert cavs[0].is_ego
            if key not in cavs[0].data:
                continue
            d = cavs[0].data[key]
            data[cavs[0].id] = d
        return data

    def gather_cav_data(self, key):
        data = {}
        for cavs in self.cav_manager.cavs:
            for cav in cavs:
                d = cav.data.get(key, {})
                if key == 'detection_local' and not cav.transformed:
                    T = cavs[0].T_g2aug @ cav.T_aug2g
                    d = copy.deepcopy(d)
                    d['preds']['box'][:, :7] = transform_boxes_3d(d['preds']['box'][:, :7], T)
                data[cav.id] = d
        return data

    def boxes_to_vis_format(self, boxes, labels, id_appendix=0):
        boxes_vis = {}
        gt_labels = labels.tolist()
        for i, box in enumerate(boxes.tolist()):
            cur_id = i + 1
            if id_appendix != 0:
                cur_id = cur_id * 10 + id_appendix
            try:
                boxes_vis[cur_id] = [gt_labels[i]] + box[:6] + [0, 0] + [box[6]]
            except:
                print('d')
        return boxes_vis

    def get_gt_boxes_as_vis_format(self, batch_idx, coor='global', successors=False):
        gt_boxes = self.gather_batch(batch_idx, f'{coor}_bboxes_3d' )
        gt_labels = self.gather_batch(batch_idx, f'{coor}_labels_3d')
        if successors and coor=='global':
            bboxes_3d_pred = self.gather_batch(batch_idx, 'bboxes_3d_pred')
        labels = {}
        successor_labels = {}
        for k in gt_boxes.keys():
            labels[k] = self.boxes_to_vis_format(gt_boxes[k], gt_labels[k])
            if successors and coor=='global' and k in bboxes_3d_pred:
                successor_labels[k] = {}
                for i, cur_preds in enumerate(bboxes_3d_pred[k]):
                    tmp_boxes = gt_boxes[k].detach().clone()
                    tmp_boxes[:, :3] = cur_preds[:, :3]
                    tmp_boxes[:, 6] = cur_preds[:, -1]
                    successor_labels[k].update(self.boxes_to_vis_format(tmp_boxes, gt_labels[k], i))
        return labels, successor_labels


    def gather_vis_data(self, batch_idx=0, keys=['points']):
        gather_dict = {}
        successors = 'global_pred_gt' in keys
        for k in keys:
            if k in ['global_labels', 'local_labels']:
                ref_coor = k.split('_')[0]
                gather_dict[f'{ref_coor}_labels'], successor_labels = (
                    self.get_gt_boxes_as_vis_format(batch_idx, ref_coor, successors))
                if successors and ref_coor=='global':
                    gather_dict['global_pred_gt'] = successor_labels
            elif k == 'global_pred_gt' or k == 'global_pred':
                continue
            elif k == 'detection' or k == 'detection_global':
                detection = self.gather_ego_data(k)
                global_pred = {}
                for cav_id, det in detection.items():
                    global_pred[cav_id] = {}
                    if 'preds' in det:
                        det = det['preds'] # todo: without nms hook, keywork preds is not removed
                    if 'box' in det and 'lbl' in det:
                        detection[cav_id]['labels'] = self.boxes_to_vis_format(det['box'], det['lbl'])
                    if 'pred' in det and det['pred'] is not None:
                        global_pred[cav_id]['labels'] = self.boxes_to_vis_format(
                            det['pred'].view(-1, 7), det['lbl'].unsqueeze(0).repeat(2, 1).view(-1))
                gather_dict['detection'] = detection
                gather_dict['global_pred'] = global_pred
            elif k == 'detection_local':
                detection = self.gather_batch(batch_idx, k)
                for cav_id, det in detection.items():
                    if len(det) == 0:
                        continue
                    if 'preds' in det:
                        det = det['preds']
                    if 'box' in det and 'lbl' in det:
                        detection[cav_id]['labels'] = self.boxes_to_vis_format(det['box'], det['lbl'])
                gather_dict['detection_local'] = detection
            else:
                gather_dict[k] = self.gather_batch(batch_idx, k, True)
        return gather_dict

    def get_vis_data_input(self, batch_idx=0, keys=None):
        """

        Parameters
        ----------
        batch_idx
        key: additional gt keys that are not standarlized in consense3d data API

        Returns
        -------

        """
        pcds = self.gather_batch(batch_idx, 'points', True)
        imgs = self.gather_batch(batch_idx, 'img', True)
        global_labels = self.get_gt_boxes_as_vis_format(batch_idx, 'global')
        local_labels = self.get_gt_boxes_as_vis_format(batch_idx, 'local')
        bboxes2d = self.gather_batch(batch_idx, 'bboxes2d', True)
        lidar2img = self.gather_batch(batch_idx, 'lidar2img', True)
        out_dict = {
            'pcds': pcds,
            'imgs': imgs,
            'bboxes2d': bboxes2d,
            'lidar2img': lidar2img,
            'global_labels': global_labels,
            'local_labels': local_labels
        }
        if keys is not None:
            for k in keys:
                out_dict[k] = self.gather_batch(batch_idx, k, True)
        return out_dict

    def get_vis_data_detection(self, batch_idx=0, keys='detection'):
        """

        Parameters
        ----------
        batch_idx: batch index
        key: the default key for detection is 'detection', customized key can also be used,
        depending on which key is used for saving detection result in the CAV data pool.

        Returns
        -------
            detection: result with boxes and labels converted to the visualizing format.
        """
        detection = self.gather_batch(batch_idx, 'detection')
        for cav_id, det in detection.items():
            detection[cav_id]['labels'] = self.boxes_to_vis_format(det['box'], det['lbl'])
        return detection

    def get_vis_data_bev(self, batch_idx=0, keys='bev'):
        return self.gather_batch(batch_idx, 'bev')

    def get_vis_data_meta(self, batch_idx=0, keys=None):
        return {
            'scenario': self.gather_batch(batch_idx, 'scenario'),
            'frame': self.gather_batch(batch_idx, 'frame')
        }

    def vis_global_data_plt(self, vis_funcs, seq_len=1):
        for func in vis_funcs:
            ax = None
            for cav in self.cav_manager.cavs[0]:
                ax = getattr(cav, func)(ax, his_len=seq_len)
            plt.savefig(f"{os.environ['HOME']}/Pictures/{func}_{seq_len}.png")
            plt.close()








