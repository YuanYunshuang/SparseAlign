import math
import os
from collections import OrderedDict
from random import random

import torch
from torch import nn
import numpy as np

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.misc import MLN
import cosense3d.modules.utils.positional_encoding as PE
from cosense3d.utils.box_utils import transform_boxes_3d
from cosense3d.utils.pclib import pose_to_transformation, tf2pose
from cosense3d.modules.losses import build_loss
from cosense3d.ops.iou3d_nms_utils import boxes_iou_bev
try:
    import g2o
    from cosense3d.modules.plugin.object_registration import GraphVertexRegistration
    from cosense3d.modules.plugin.pose_graph_optim import PoseGraphOptimization2D
except:
    g2o = None
    GraphVertexRegistration = None
    PoseGraphOptimization2D = None



class PseudoPoseAlign(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ego_poses, coop_feats, ego_dets, **kwargs):
        results = []
        for ego_pose, coop_dict, ego_det in zip(
                ego_poses, coop_feats, ego_dets):
            if len(coop_dict) == 0:
                results.append(None)
                continue

            T_g2e = ego_pose.inverse()
            res = []
            for cpm in coop_dict.values():
                T_c2e = T_g2e @ cpm['pose']
                res.append(T_c2e)
            results.append(res)
        return {self.scatter_keys[0]: results}


class NbrBasedPoseAlign(BaseModule):
    def __init__(self,
                 disable_align=False,
                 err_while_training=False,
                 ego_pc_range=[-100, -40, 100, 40],
                 coop_pc_range=[-100, -40, 100, 40],
                 **kwargs):
        super().__init__(**kwargs)
        self.disable_align = disable_align
        self.err_while_training = err_while_training
        self.ego_pc_range = ego_pc_range
        self.coop_pc_range = coop_pc_range
        if not disable_align and GraphVertexRegistration is not None:
            self.gvr = GraphVertexRegistration(8)
            path = os.path.abspath(__file__).split('cosense3d')[0] + "assets/model_nbr8.pth"
            self.gvr.load_state_dict(torch.load(path))

        self.cnt = 0

    def forward(self, ego_poses, coop_feats, ego_dets, **kwargs):
        if self.training or self.disable_align:
            return self.pseudo_align(ego_poses, coop_feats, ego_dets, **kwargs)
        else:
            return self.align(ego_poses, coop_feats, ego_dets, **kwargs)

    def pseudo_align(self, ego_poses, coop_feats, ego_dets, **kwargs):
        results = []
        for ego_pose, coop_dict, ego_det in zip(
                ego_poses, coop_feats, ego_dets):
            if len(coop_dict) == 0:
                results.append(None)
                continue

            T_g2e = ego_pose.inverse()
            res = []
            for cpm in coop_dict.values():
                T_c2e = T_g2e @ cpm['pose']
                if self.err_while_training:
                    T_c2e[:2, -1] *= torch.randn(2, device=T_c2e.device) * 0.2
                res.append(T_c2e)
            results.append(res)
        return {self.scatter_keys[0]: results}

    def align(self, ego_poses, coop_feats, ego_dets, **kwargs):
        results = []
        self.cnt += 1
        for ego_pose, coop_dict, ego_det in zip(
                ego_poses, coop_feats, ego_dets):
            assert (ego_det['preds']['idx'] == 0).all(), 'only support batch size one.'
            if len(coop_dict) == 0 or len(ego_det['preds']['box']) <= 1:
                results.append(None)
                continue

            # get bboxes for matching
            boxes = [ego_det['preds']['box']]
            box_masks = []
            poses = [ego_pose]
            poses_all = [ego_pose]
            cav_align_mask = []
            for cpm in coop_dict.values():
                coop_boxes = cpm['preds']['box']
                poses_all.append(cpm['pose'])
                masks = self.filter_boxes_range(coop_boxes, boxes[0], cpm['pose'], ego_pose)
                n_valid = min(masks[0].sum(), masks[1].sum())
                if n_valid > 1:
                    cav_align_mask.append(True)
                    boxes.append(coop_boxes)
                    poses.append(cpm['pose'])
                    box_masks.append(masks)
                else:
                    cav_align_mask.append(False)

            new_poses = [x.clone() for x in poses]
            # feats = [self.gvr.get_nbrhood_features(x) for x in boxes]

            # box_masks = [[x[0].cpu().numpy(), x[1].cpu().numpy()] for x in box_masks]
            # new_poses = [x.cpu().numpy() for x in [poses[0], poses[1]]]
            # feats = [self.gvr.get_nbrhood_features(x) for x in [boxes[0], boxes[1]]]
            # boxes_np = [x.cpu().numpy() for x in [boxes[0], boxes[1]]]
            for _ in range(1):
                new_poses = self.refine_single(boxes, new_poses, box_masks)

            new_rel_poses = self.pose_abs_to_rel(new_poses, ego_pose.device)
            rel_poses = self.pose_abs_to_rel(poses, ego_pose.device)
            rel_poses_corr = self.validate_correction(rel_poses, new_rel_poses, boxes)

            # boxes_np = [x.cpu().numpy() for x in boxes]
            # self.vis_fusion_result(rel_poses_corr, boxes_np)

            ptr = 0
            rel_poses_ = []
            for flag, p in zip(cav_align_mask, poses_all):
                if flag:
                    x = rel_poses_corr[ptr]
                    if x is None:
                        rel_poses_.append(p)
                    else:
                        rel_poses_.append(x)
                    ptr += 1
                else:
                    rel_poses_.append(p)

            results.append(rel_poses_)
        return {self.scatter_keys[0]: results}

    def filter_boxes_range(self, coop_boxes, ego_boxes, coop_pose, ego_pose):
        ego_boxes_c = transform_boxes_3d(ego_boxes[:, :7], coop_pose.inverse() @ ego_pose)
        mask_e = (ego_boxes_c[:, 0] > self.coop_pc_range[0]) & (ego_boxes_c[:, 0] < self.coop_pc_range[2]) & \
            (ego_boxes_c[:, 1] > self.coop_pc_range[1]) & (ego_boxes_c[:, 1] < self.coop_pc_range[3])
        coop_boxes_e = transform_boxes_3d(coop_boxes[:, :7], ego_pose.inverse() @ coop_pose)
        mask_c = (coop_boxes_e[:, 0] > self.ego_pc_range[0]) & (coop_boxes_e[:, 0] < self.ego_pc_range[2]) & \
            (coop_boxes_e[:, 1] > self.ego_pc_range[1]) & (coop_boxes_e[:, 1] < self.ego_pc_range[3])

        # ego_boxes_w = transform_boxes_3d(ego_boxes[:, :7], ego_pose)
        # coop_boxes_w = transform_boxes_3d(coop_boxes[:, :7], coop_pose)
        # ego_boxes_w_s = ego_boxes_w[mask_e]
        # coop_boxes_w_s = coop_boxes_w[mask_c]
        # ego_range = transform_boxes_3d(torch.tensor([[0, 0, 0, 200, 80, 1, 0]]).to(ego_pose.device), ego_pose)
        # coop_range = transform_boxes_3d(torch.tensor([[50, 0, 0, 100, 80, 1, 0]]).to(coop_pose.device), coop_pose)
        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        #
        # r = ego_pose.cpu().numpy()
        # pc_range = [r[0, 3] - 103, r[1, 3] - 103,
        #             r[0, 3] + 103, r[1, 3] + 103]
        #
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     boxes_gt=ego_boxes_w,
        #     bbox_gt_c='grey',
        #     boxes_pred=coop_boxes_w,
        #     bbox_pred_c='grey',
        #     return_ax=True,
        # )
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     boxes_gt=ego_boxes_w_s,
        #     boxes_pred=coop_boxes_w_s,
        #     return_ax=True,
        #     ax=ax
        # )
        #
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     boxes_gt=ego_range,
        #     boxes_pred=coop_range,
        #     return_ax=True,
        #     ax=ax
        # )
        # ax.plot(r[0, 3], r[1, 3], "*g", markersize=10)
        # r = coop_pose.cpu().numpy()
        # ax.plot(r[0, 3], r[1, 3], "*r", markersize=10)
        # plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
        # plt.close()

        return mask_e, mask_c

    def refine_single(self, boxes, poses, box_masks):
        boxes_world = [transform_boxes_3d(b[:, :7],
                    torch.from_numpy(p).to(b.device) if isinstance(p, np.ndarray) else p)
                       for b, p in zip(boxes, poses)]
        cluster_inds = [np.arange(len(boxes[0]))]
        cluster_idx = len(cluster_inds[0])
        for box, boxw, pose, masks in zip(boxes[1:], boxes_world[1:], poses[1:], box_masks):
            row_ind, col_ind, _ = self.gvr.match_pair(boxes_world[0][masks[0]], boxw[masks[1]])
            row_ind = np.arange(len(boxes_world[0]))[masks[0].tolist()][row_ind]
            col_ind = np.arange(len(boxw))[masks[1].tolist()][col_ind]
            coop_cluster_inds = np.zeros(len(box), dtype=int) - 1
            coop_cluster_inds[col_ind] = cluster_inds[0][row_ind]
            new_mask = coop_cluster_inds < 0
            cluster_idx_next = cluster_idx + new_mask.sum()
            coop_cluster_inds[new_mask] = np.arange(cluster_idx, cluster_idx_next)
            cluster_inds.append(coop_cluster_inds)
            cluster_idx = cluster_idx_next
            # self.vis_match_result(boxes_world[0], boxw, row_ind, col_ind, poses[0][:2, 3].tolist())

        new_poses = self.optimize(boxes_world, boxes, cluster_inds, poses)
        # self.vis_correction_result(poses, new_poses, boxes)
        return new_poses

    def optimize(self, boxes_world, boxes_local, cluster_inds, poses):
        boxes_world = [x.cpu().numpy() for x in boxes_world]
        boxes_local = [x.cpu().numpy() for x in boxes_local]
        poses = [x.cpu().numpy() for x in poses]
        N = len(boxes_world)
        pgo = PoseGraphOptimization2D()
        # add agent to vertices
        for i in range(N):
            p = tf2pose(poses[i])
            v_pose = g2o.SE2([p[0], p[1], p[5]])
            fixed = True if i == 0 else False
            pgo.add_vertex(id=i, pose=v_pose, fixed=fixed)

        ptr = N
        agent_inds = np.concatenate([np.zeros_like(x[:, 0]) + i
                                     for i, x in enumerate(boxes_world)],
                                    axis=0).astype(int)
        boxes_world = np.concatenate(boxes_world, axis=0)
        boxes_local = np.concatenate(boxes_local, axis=0)
        cluster_inds = np.concatenate(cluster_inds, axis=0)
        uniq_inds = np.unique(cluster_inds)
        info = np.identity(2, dtype=np.float64)

        for idx in uniq_inds:
            mask = cluster_inds == idx
            cur_boxes_local = boxes_local[mask]
            cur_boxes_world = boxes_world[mask]
            cur_agent_inds = agent_inds[mask]

            if len(cur_boxes_local) == 0:
                continue

            # add cluster vertex
            pgo.add_vertex(id=ptr, pose=cur_boxes_world[0, :2], fixed=False, SE2=False)
            # for v_pose in cur_boxes_world:
            #     pgo.add_vertex(id=idx, pose=v_pose[:2], fixed=False, SE2=False)
            # add agent-object edges
            for i in range(len(cur_boxes_local)):
                agent_idx = cur_agent_inds[i]
                pgo.add_edge(vertices=[agent_idx, ptr], measurement=cur_boxes_local[i, :2], information=info, SE2=False)
            ptr += 1

        pgo.optimize(1000)
        refined_poses = [pgo.get_pose(i) for i in range(N)]
        refined_poses = [pose_to_transformation([p[0], p[1], 0, 0, 0, p[2]]) for p in refined_poses]
        return refined_poses

    def pose_abs_to_rel(self, poses, device):
        T_c2e = []
        is_numpy = isinstance(poses[0], np.ndarray)
        T_g2e = np.linalg.inv(poses[0]) if is_numpy else poses[0].inverse()
        for p in poses[1:]:
            if is_numpy:
                T = torch.from_numpy(T_g2e @ p).float().to(device)
            else:
                T = T_g2e @ p
            T_c2e.append(T)
        return T_c2e

    def vis_match_result(self, ego_boxes, coop_boxes, row_ind, col_ind, pose):
        matched_ego_boxes = ego_boxes[row_ind, :7].cpu().numpy()
        matched_coop_boxes = coop_boxes[col_ind, :7].cpu().numpy()

        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        pc_range = [pose[0] - 100, pose[1] - 100,
                    pose[0] + 100,pose[1] + 100]
        ax = draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_gt=matched_ego_boxes,
            return_ax=True
        )
        ax = draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_pred=matched_coop_boxes,
            return_ax=True,
            ax=ax
        )

        for b1, b2 in zip(matched_ego_boxes, matched_coop_boxes):
            ax.plot([b1[0], b2[0]], [b1[1], b2[1]], color='black')

        plt.savefig(f"{os.environ['HOME']}/Downloads/match_res.jpg")
        plt.close()

    def vis_correction_result(self, poses, refined_poses, boxes):
        poses = [x.cpu().numpy() for x in poses]
        boxes = [x.cpu().numpy() for x in boxes]
        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        fig = plt.figure(figsize=(12, 6))
        axs = fig.subplots(1, 2)
        colors = ['g', 'r', 'b']
        r = refined_poses[0]
        pc_range = [r[0, 3] - 100, r[1, 3] - 100,
                    r[0, 3] + 100, r[1, 3] + 100]
        # before correction
        for i, b in enumerate(boxes):
            p = poses[i]
            b = transform_boxes_3d(b[:, :7], p, mode=7)
            axs[0] = draw_points_boxes_plt(
                pc_range=pc_range,
                boxes_gt=b,
                bbox_gt_c=colors[i],
                return_ax=True,
                ax=axs[0]
            )
        # after correction
        for i, b in enumerate(boxes):
            p = refined_poses[i]
            # p = pose_to_transformation([p[0], p[1], 0, 0, 0, p[2]])
            # p = p @ poses[i]
            b = transform_boxes_3d(b[:, :7], p, mode=7)
            axs[1] = draw_points_boxes_plt(
                pc_range=pc_range,
                boxes_gt=b,
                bbox_gt_c=colors[i],
                return_ax=True,
                ax=axs[1]
            )

        plt.savefig(f"{os.environ['HOME']}/Downloads/corr_res.jpg")
        # plt.savefig(f"/home/yys/Downloads/match_res/{self.cnt}.jpg")
        plt.close()

    def validate_correction(self, rel_poses, new_rel_poses, boxes):
        coop_boxes = [transform_boxes_3d(b[:, :7], p, mode=7) for p, b in zip(rel_poses, boxes[1:])]
        new_coop_boxes = [transform_boxes_3d(b[:, :7], p, mode=7) for p, b in zip(new_rel_poses, boxes[1:])]
        out_poses = []
        for i, b in enumerate(coop_boxes):
            ious = boxes_iou_bev(b, boxes[0][:, :7])
            ious_new = boxes_iou_bev(new_coop_boxes[i], boxes[0][:, :7])
            iou_max = ious.max(dim=-1).values
            iou_max_new = ious_new.max(dim=-1).values
            n_match = (iou_max > 0.3).sum()
            n_match_new = (iou_max > 0.3).sum()
            iou_mean = iou_max[iou_max > 0].mean()
            iou_mean_new = iou_max_new[iou_max > 0].mean()
            if n_match_new > n_match or iou_mean_new > iou_mean:
                out_poses.append(new_rel_poses[i])
            elif iou_mean < 0.3 or n_match < 2:
                out_poses.append(None)
            else:
                out_poses.append(rel_poses[i])
        return out_poses

    def vis_fusion_result(self, rel_poses, boxes):
        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        colors = ['g', 'r', 'b', 'c', 'k', 'y', 'orange']

        ax = draw_points_boxes_plt(
            pc_range=100,
            boxes_gt=boxes[0],
            bbox_gt_c=colors[0],
            return_ax=True,
        )
        for i, b in enumerate(boxes[1:]):
            p = rel_poses[i]
            if p is None:
                continue
            box = transform_boxes_3d(b[:, :7], p.cpu().numpy(), mode=7)
            ax = draw_points_boxes_plt(
                pc_range=100,
                boxes_gt=box,
                bbox_gt_c=colors[i+1],
                return_ax=True,
                ax=ax
            )

        plt.savefig(f"{os.environ['HOME']}/Downloads/corr_res.jpg")
        # plt.savefig(f"/home/yys/Downloads/opv2v/match_result/{self.cnt}.jpg")
        plt.close()


class CoAlign(BaseModule):
    def __init__(self, disable_align=False, **kwargs):
        super().__init__(**kwargs)
        self.disable_align = disable_align

    def forward(self, ego_poses, coop_feats, ego_dets, **kwargs):
        if self.training or self.disable_align:
            return self.pseudo_align(ego_poses, coop_feats, ego_dets, **kwargs)
        else:
            return self.align(ego_poses, coop_feats, ego_dets, **kwargs)

    def pseudo_align(self, ego_poses, coop_feats, ego_dets, **kwargs):
        results = []
        for ego_pose, coop_dict, ego_det in zip(
                ego_poses, coop_feats, ego_dets):
            if len(coop_dict) == 0:
                results.append(None)
                continue

            T_g2e = ego_pose.inverse()
            res = []
            for cpm in coop_dict.values():
                T_c2e = T_g2e @ cpm['pose']
                res.append(T_c2e)
            results.append(res)
        return {self.scatter_keys[0]: results}

    def align(self, ego_poses, coop_feats, ego_dets, **kwargs):
        results = []
        for ego_pose, coop_dict, ego_det in zip(
                ego_poses, coop_feats, ego_dets):
            assert (ego_det['preds']['idx'] == 0).all(), 'only support batch size one.'
            if len(coop_dict) == 0 or len(ego_det['preds']['box']) <= 1:
                results.append(None)
                continue

            # get bboxes for matching
            boxes = [ego_det['preds']['box']]
            poses = [ego_pose]
            cav_align_mask = []
            for cpm in coop_dict.values():
                if len(cpm['preds']['box']) <= 1:
                    cav_align_mask.append(False)
                else:
                    cav_align_mask.append(True)
                    boxes.append(cpm['preds']['box'])
                    poses.append(cpm['pose'])

            new_poses = [x.cpu().numpy() for x in poses]
            boxes_np = [x.cpu().numpy() for x in boxes]
            new_poses = self.refine_single(boxes_np, new_poses)
            rel_poses = self.pose_abs_to_rel(new_poses, ego_pose.device)
            ptr = 0
            rel_poses_ = []
            for flag in cav_align_mask:
                if flag:
                    rel_poses_.append(rel_poses[ptr])
                    ptr += 1
                else:
                    rel_poses_.append(None)
            results.append(rel_poses_)
        return {self.scatter_keys[0]: results}

    def refine_single(self, boxes, poses):
        boxes_world = [transform_boxes_3d(b[:, :7], p) for b, p in zip(boxes, poses)]
        cluster_dict, nums = self.clustering(boxes_world)

        if cluster_dict is None:
            new_poses = poses
        else:
            new_poses = self.optimize(boxes_world, boxes, cluster_dict, nums, poses)

        # self.vis_correction_result(poses, new_poses, boxes)
        return new_poses

    def clustering(self, boxes_world: list,
                   thres: float=1.5,
                   yaw_var_thres: float=0.2,
                   abandon_hard_cases=True,
                   drop_hard_boxes=True):
        N = len(boxes_world)
        box_idx_to_agent = []
        pred_len = []
        for i in range(N):
            box_idx_to_agent += [i] * len(boxes_world[i])
            pred_len.append(len(boxes_world[i]))

        boxes_world_cat = np.concatenate(boxes_world, axis=0)
        pred_center_allpair_dist = self.all_pair_l2(boxes_world_cat[:, :3],
                                                    boxes_world_cat[:, :3])  # [sum(pred_box), sum(pred_box)]
        # let pair from one vehicle be max distance
        MAX_DIST = 10000
        cum = 0
        for i in range(N):
            pred_center_allpair_dist[cum: cum + pred_len[i], cum: cum + pred_len[i]] = MAX_DIST  # do not include itself
            cum += pred_len[i]

        cluster_id = N  # let the vertex id of object start from N
        cluster_dict = OrderedDict()
        remain_box = set(range(cum))

        for box_idx in range(cum):
            if box_idx not in remain_box:  # already assigned
                continue

            within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
            within_thres_idx_list = within_thres_idx_tensor.tolist()

            if len(within_thres_idx_list) == 0:  # if it's a single box
                continue

            # start from within_thres_idx_list, find new box added to the cluster
            explored = [box_idx]
            unexplored = [idx for idx in within_thres_idx_list if idx in remain_box]

            while unexplored:
                idx = unexplored[0]
                within_thres_idx_tensor = (pred_center_allpair_dist[box_idx] < thres).nonzero()[0]
                within_thres_idx_list = within_thres_idx_tensor.tolist()
                for newidx in within_thres_idx_list:
                    if (newidx not in explored) and (newidx not in unexplored) and (newidx in remain_box):
                        unexplored.append(newidx)
                unexplored.remove(idx)
                explored.append(idx)

            if len(explored) == 1:  # it's a single box, neighbors have been assigned
                remain_box.remove(box_idx)
                continue

            cluster_box_idxs = explored

            cluster_dict[cluster_id] = OrderedDict()
            cluster_dict[cluster_id]['box_idx'] = [idx for idx in cluster_box_idxs]
            cluster_dict[cluster_id]['box_center_world'] = [boxes_world_cat[:, :3][idx] for idx in
                                                            cluster_box_idxs]  # coordinate in world, [3,]
            cluster_dict[cluster_id]['box_yaw'] = [boxes_world_cat[:, 6][idx] for idx in cluster_box_idxs]

            yaw_var = np.var(cluster_dict[cluster_id]['box_yaw'])
            cluster_dict[cluster_id]['box_yaw_varies'] = yaw_var > yaw_var_thres
            cluster_dict[cluster_id]['active'] = True

            landmark = boxes_world_cat[box_idx][[0, 1, 6]]
            cluster_dict[cluster_id]['landmark'] = landmark  # [x, y, yaw]

            cluster_id += 1
            for idx in cluster_box_idxs:
                remain_box.remove(idx)

        vertex_num = cluster_id
        agent_num = N
        landmark_num = cluster_id - N

        if abandon_hard_cases:
            # case1: object num is smaller than 3
            if landmark_num <= 3:
                return None, None

            # case2: more than half of the landmarks yaw varies
            yaw_varies_cnt = sum([cluster_dict[i]["box_yaw_varies"] for i in range(agent_num, vertex_num)])
            if yaw_varies_cnt >= 0.5 * landmark_num:
                return None, None

        ########### drop hard boxes ############

        if drop_hard_boxes:
            for landmark_id in range(agent_num, vertex_num):
                if cluster_dict[landmark_id]['box_yaw_varies']:
                    cluster_dict[landmark_id]['active'] = False

        return cluster_dict, {'agent_num': agent_num, 'landmark_num': landmark_num, 'vertex_num': vertex_num}


    @staticmethod
    def all_pair_l2(A, B):
        """ All pair L2 distance for A and B
        Args:
            A : np.ndarray
                shape [N_A, D]
            B : np.ndarray
                shape [N_B, D]
        Returns:
            C : np.ndarray
                shape [N_A, N_B]
        """
        TwoAB = 2 * A @ B.T  # [N_A, N_B]
        C = np.sqrt(
            np.sum(A * A, 1, keepdims=True).repeat(TwoAB.shape[1], axis=1) \
            + np.sum(B * B, 1, keepdims=True).T.repeat(TwoAB.shape[0], axis=0) \
            - TwoAB
        )
        return C

    def optimize(self, boxes_world, boxes_local, cluster_dict, nums, poses, landmark_SE2=False):
        agent_inds = np.concatenate([np.zeros_like(x[:, 0]) + i
                                     for i, x in enumerate(boxes_world)],
                                    axis=0).astype(int)
        N = len(boxes_world)
        boxes_world = np.concatenate(boxes_world, axis=0)
        boxes_local = np.concatenate(boxes_local, axis=0)

        pgo = PoseGraphOptimization2D()
        # add agent to vertices
        for i in range(N):
            p = tf2pose(poses[i])
            v_pose = g2o.SE2([p[0], p[1], p[5]])
            fixed = True if i == 0 else False
            pgo.add_vertex(id=i, pose=v_pose, fixed=fixed)

        for landmark_id in range(nums['agent_num'], nums['vertex_num']):
            v_id = landmark_id
            landmark = cluster_dict[landmark_id]['landmark']  # (3,) or (2,)
            v_pose = landmark if landmark_SE2 else landmark[:2]
            # Add object to vertexs
            pgo.add_vertex(id=v_id, pose=v_pose, fixed=False, SE2=landmark_SE2)

            # add agent-object edges
            if not cluster_dict[landmark_id]['active']:
                continue
            for box_idx in cluster_dict[landmark_id]['box_idx']:
                agent_id = agent_inds[box_idx]
                if landmark_SE2:
                    e_pose = g2o.SE2(boxes_local[box_idx][[0, 1, 6]].astype(np.float64))
                    info = np.identity(3, dtype=np.float64)
                else:
                    e_pose = boxes_local[box_idx][[0, 1]].astype(np.float64)
                    info = np.identity(2, dtype=np.float64)

                pgo.add_edge(vertices=[agent_id, v_id], measurement=e_pose, information=info, SE2=False)

        pgo.optimize(1000)
        refined_poses = [pgo.get_pose(i) for i in range(N)]
        refined_poses = [pose_to_transformation([p[0], p[1], 0, 0, 0, p[2]]) for p in refined_poses]
        return refined_poses

    def pose_abs_to_rel(self, poses, device):
        T_c2e = []
        T_g2e = np.linalg.inv(poses[0])
        for p in poses[1:]:
            T_c2e.append(torch.from_numpy(T_g2e @ p).float().to(device))
        return T_c2e

    def vis_match_result(self, ego_boxes, coop_boxes, row_ind, col_ind, pose):
        matched_ego_boxes = ego_boxes[row_ind, :7]
        matched_coop_boxes = coop_boxes[col_ind, :7]

        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        pc_range = [pose[0] - 100, pose[1] - 100,
                    pose[0] + 100,pose[1] + 100]
        ax = draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_gt=matched_ego_boxes,
            return_ax=True
        )
        ax = draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_pred=matched_coop_boxes,
            return_ax=True,
            ax=ax
        )

        for b1, b2 in zip(matched_ego_boxes, matched_coop_boxes):
            ax.plot([b1[0], b2[0]], [b1[1], b2[1]], color='black')

        plt.savefig(f"{os.environ['HOME']}/Downloads/match_res.jpg")
        plt.close()

    def vis_correction_result(self, poses, refined_poses, boxes):
        from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        fig = plt.figure(figsize=(12, 6))
        axs = fig.subplots(1, 2)
        colors = ['g', 'r', 'b']
        r = refined_poses[0]
        pc_range = [r[0, 3] - 100, r[1, 3] - 100,
                    r[0, 3] + 100, r[1, 3] + 100]
        # before correction
        for i, b in enumerate(boxes):
            p = poses[i]
            b = transform_boxes_3d(b[:, :7], p, mode=7)
            axs[0] = draw_points_boxes_plt(
                pc_range=pc_range,
                boxes_gt=b,
                bbox_gt_c=colors[i],
                return_ax=True,
                ax=axs[0]
            )
        # before correction
        for i, b in enumerate(boxes):
            p = refined_poses[i]
            # p = pose_to_transformation([p[0], p[1], 0, 0, 0, p[2]])
            # p = p @ poses[i]
            b = transform_boxes_3d(b[:, :7], p, mode=7)
            axs[1] = draw_points_boxes_plt(
                pc_range=pc_range,
                boxes_gt=b,
                bbox_gt_c=colors[i],
                return_ax=True,
                ax=axs[1]
            )

        plt.savefig(f"{os.environ['HOME']}/Downloads/corr_res.jpg")
        plt.close()


class PoseAlign(BaseModule):
    def __init__(self, in_channels, pc_range, loss=None, **kwargs):
        super().__init__(**kwargs)
        self.topk = 64
        self.in_channels = in_channels
        self.rotation_pe = MLN(9 * 12, f_dim=in_channels)
        self.pc_range = nn.Parameter(torch.Tensor(pc_range), requires_grad=False)
        self._init_layers(in_channels)
        if loss is not None:
            self.loss_cls = build_loss(**loss)

    def _init_layers(self, in_channels):
        self.pos_embedding = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.LayerNorm(in_channels // 2),
            nn.Linear(in_channels // 2, in_channels // 2),
        )

        self.feat_embedding = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.LayerNorm(in_channels // 2),
            nn.Linear(in_channels // 2, in_channels // 2),
        )
        self.nbr_pool_weights = nn.Linear(in_channels, 1)

        self.merge_local_nbr_feat = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.LayerNorm(in_channels)
        )

    def forward(self, ego_poses, ego_feats, coop_feats, ego_dets, **kwargs):
        results = []
        self.results = []
        for ego_pose, ego_dict, coop_dict, ego_det in zip(
                ego_poses, ego_feats, coop_feats, ego_dets):
            if len(coop_dict) == 0:
                results.append(None)
                continue

            # fuse coop to ego
            for cpfeat in coop_dict.values():
                res = self.align(ego_pose, ego_dict, ego_det, cpfeat[self.gather_keys[1]])
                self.results.append(res)
                results.append(None)
        return {self.scatter_keys[0]: results}

    def align(self, ego_pose, ego_dict, ego_det, coop_dict):
        scores = ego_det['all_cls_scores'][-1]
        scores = scores[..., min(scores.shape[-1] - 1, 1):].topk(1, dim=-1).values[..., 0]
        mask = scores > 0.3
        if mask.sum() > 512:
            topk, mask = torch.topk(scores, 1024)
        ego_ctr = ego_pose[:2, -1]
        ego_pts, ego_feat_emb = self.nbrhood_embedding(
            ego_dict['ref_pts'][mask],
            ego_dict['outs_dec'][:, -1][mask],
            ego_pose
        )
        coop_pts, coop_feat_emb = self.nbrhood_embedding(
            coop_dict['ref_pts'],
            coop_dict['outs_dec'][:, -1],
            coop_dict['pose']
        )

        diff = ego_pts[:, :2].unsqueeze(1) - coop_pts[:, :2].unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        nbr_dist, nbr_inds = torch.topk(dist, k=self.topk, dim=1, largest=False)
        mask = nbr_dist.min(dim=-1)[0] < 1.6
        nbr_dist, nbr_inds = nbr_dist[mask], nbr_inds[mask]
        ego_feat_emb = ego_feat_emb[mask]
        coop_feat_emb_nbr = coop_feat_emb[nbr_inds.flatten()].view(*nbr_inds.shape, coop_feat_emb.shape[-1])

        dotprod = torch.bmm(coop_feat_emb_nbr, ego_feat_emb.unsqueeze(2)) / math.sqrt(self.in_channels)
        similarity = dotprod.squeeze(-1)
        # similarity = torch.softmax(dotprod, dim=1).squeeze(-1)

        gt_similarity = torch.zeros_like(similarity)
        match_mask = nbr_dist < 1.6
        gt_similarity[match_mask] = 1 - nbr_dist[match_mask] / 1.6

        # self.vis(ego_pts, coop_pts, nbr_inds, nbr_dist)
        return similarity, gt_similarity

    def transform(self, coor, feat, T):
        coor = (T @ torch.cat(
            [coor, torch.ones_like(coor[:, :1])
             ], dim=1).T)[:3].T
        Remb = PE.nerf_positional_encoding(T[:3, :3].reshape(1, -1))
        feat = self.rotation_pe(feat, Remb)
        return coor, feat

    def nbrhood_embedding(self, pts, feat, pose):
        pts, feat = self.transform(pts, feat, pose)

        diff = pts[:, :2].unsqueeze(1) - pts[:, :2].unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        topk = min(self.topk, pts.shape[0])
        min_dist, min_inds = torch.topk(dist, k=topk, dim=1, largest=False)
        diff_pos = torch.gather(diff, dim=1, index=min_inds.unsqueeze(2).repeat(1, 1, 2))
        diff_pos_emb = (diff_pos - self.pc_range[:2]) / (self.pc_range[3:5] - self.pc_range[:2])
        diff_pos_emb = self.pos_embedding(PE.pos2posemb2d(diff_pos_emb, self.in_channels // 2))

        nbr_feat = feat[min_inds.flatten()].view(*min_inds.shape, feat.shape[-1])
        diff_feat = feat.unsqueeze(1) - nbr_feat
        diff_feat_emb = self.feat_embedding(diff_feat)

        nbrhood_emb = torch.cat([diff_feat_emb, diff_pos_emb], dim=-1)

        pool_weights = self.nbr_pool_weights(nbrhood_emb)
        nbrhood_emb = (pool_weights.softmax(dim=1) * nbrhood_emb).sum(dim=1)

        feat_emb = self.merge_local_nbr_feat(feat + nbrhood_emb)
        return pts, feat_emb

    def loss(self, *args, **kwargs):
        if len(self.results) == 0:
            loss = torch.tensor(0)
        else:
            pred = torch.cat([x[0] for x in self.results], dim=0).view(-1, 1)
            gt = torch.cat([x[1] for x in self.results], dim=0).view(-1)

            pos = torch.where(gt > 0)[0]
            neg = torch.where(gt == 0)[0]
            n_neg = max(10, len(pos))
            neg = neg[torch.randperm(len(neg))][:n_neg]
            cared = torch.cat([pos, neg], dim=0)

            loss = self.loss_cls(pred[cared], gt[cared])
        return {'align_loss': loss}

    def vis(self, ego_pts, coop_pts, nbr_inds, nbr_dist):
        import matplotlib.pyplot as plt
        plt.gca().set_aspect('equal')
        ego_pts = ego_pts.cpu().numpy()
        coop_pts = coop_pts[nbr_inds.flatten()].view(*nbr_inds.shape, 3).cpu().numpy()
        for i, pts in enumerate(coop_pts):
            if nbr_dist[i].min() > 2:
                continue
            plt.plot(ego_pts[:, 0], ego_pts[:, 1], 'b.')

            for p in pts:
                plt.plot([ego_pts[i, 0], p[0]], [ego_pts[i, 1], p[1]], 'r', linewidth=1)
            plt.savefig("/home/yuan/Downloads/tmp.jpg")
            plt.close()

        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        # pc_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
        # pc_range[0] += ego_pose[0, -1].item()
        # pc_range[1] += ego_pose[1, -1].item()
        # pc_range[3] += ego_pose[0, -1].item()
        # pc_range[4] += ego_pose[1, -1].item()
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     points=ego_pts.cpu().numpy(),
        #     points_c='blue',
        #     return_ax=True
        # )
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     points=coop_pts.cpu().numpy(),
        #     points_c='r',
        #     return_ax=True,
        #     ax=ax
        # )
        # # ax.plot(pts[0], pts[1], 'r.', markersize=1)
        # plt.savefig("/home/yys/Downloads/tmp.jpg")
        # plt.close()
        # print("d")
        #
        # pc_range = [-144, -41.6, -3.0, 144, 41.6, 1.0]
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     points=ego_dict['ref_pts'][mask].cpu().numpy(),
        #     points_c='blue',
        #     return_ax=True
        # )
        # ax = draw_points_boxes_plt(
        #     pc_range=pc_range,
        #     points=coop_pts.cpu().numpy(),
        #     points_c='r',
        #     return_ax=True,
        #     ax=ax
        # )
        # plt.savefig("/home/yys/Downloads/tmp1.jpg")
        # plt.close()

        