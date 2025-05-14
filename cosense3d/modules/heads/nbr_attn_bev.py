import torch

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.common import pad_r, linear_last, cat_coor_with_idx
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses.edl import edl_mse_loss, evidence_to_conf_unc
from cosense3d.modules.utils.nbr_attn import NeighborhoodAttention


class NbrAttentionBEV(BaseModule):
    def __init__(self,
                 data_info,
                 in_dim,
                 stride,
                 annealing_step,
                 sampling,
                 target_assigner=None,
                 class_names_each_head=None,
                 **kwargs):
        super(NbrAttentionBEV, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.class_names_each_head = class_names_each_head
        self.stride = stride
        self.annealing_step = annealing_step
        self.sampling = sampling
        for k, v in data_info.items():
            setattr(self, k, v)
        update_me_essentials(self, data_info, self.stride)

        self.nbr_attn = NeighborhoodAttention(emb_dim=in_dim)
        self.reg_layer = linear_last(in_dim, 32, 2, bias=True)

        if class_names_each_head is not None:
            from cosense3d.model.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(target_assigner,
                                               class_names_each_head)

    def forward(self, stensor_list, **kwargs):
        coor, feat, ctr = self.format_input(stensor_list)
        centers = indices2metric(coor, self.voxel_size)
        reference_points = self.generate_reference_points(centers)
        out = self.nbr_attn(feat, coor, reference_points, len(stensor_list))
        reg = self.reg_layer(out)
        conf, unc = evidence_to_conf_unc(reg.relu())

        out_dict = {
            'center': centers,
            'reg': reg,
            'conf': conf,
            'unc': unc
        }

        return self.format_output(out_dict, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['center'][:, 0] == i
            output_new['center'].append(output['center'][mask, 1:])
            output_new['reg'].append(output['reg'][mask])
            output_new['conf'].append(output['conf'][mask])
            output_new['unc'].append(output['unc'][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def generate_reference_points(self, centers):
        if self.training:
            reference_points = centers[torch.rand_like(centers[:, 0]) > 0.5]
        else:
            reference_points = centers
        noise = torch.rand_like(reference_points[:, 1:]) * self.voxel_size[0] * self.stride
        reference_points[:, 1:] = reference_points[:, 1:] + noise
        return reference_points

    def loss(self, batch_list, gt_boxes, gt_labels, **kwargs):
        tgt_pts, tgt_label, valid = self.get_tgt(batch_list, gt_boxes, gt_labels, **kwargs)
        epoch_num = kwargs.get('epoch', 0)
        reg = self.cat_data_from_list(batch_list, 'reg')
        loss_dict = edl_mse_loss(preds=reg[valid],
                                       tgt=tgt_label,
                                       n_cls=2,
                                       temp=epoch_num,
                                       annealing_step=self.annealing_step,
                                       model_label='bev')
        return loss_dict

    @torch.no_grad()
    def get_tgt(self, batch_list, gt_boxes, gt_labels, **kwargs):
        epoch_num = kwargs.get('epoch', 0)
        B = len(batch_list)
        tgt_pts = self.cat_data_from_list(batch_list, 'center', pad_idx=True)
        boxes = self.cat_data_from_list(gt_boxes, pad_idx=True).clone()
        boxes[:, 3] = 0
        pts = pad_r(tgt_pts)
        try:
            _, box_idx_of_pts = points_in_boxes_gpu(
                pts, boxes, batch_size=B
            )
            boxes[:, 4:6] *= 2
            _, box_idx_of_pts2 = points_in_boxes_gpu(
                pts, boxes, batch_size=B
            )
        except:
            print(boxes.shape)
            print(pts.shape)
        # set area B: dense neg as -1 for down-sampling, differentiate from area C: sparse neg.
        tgt_label = - (box_idx_of_pts >= 0).int()
        tgt_label[box_idx_of_pts >= 0] = 1

        n_sam = len(boxes) * 50
        if self.sampling['annealing']:
            annealing_ratio = epoch_num / self.annealing_step
            n_sam = n_sam + annealing_ratio * len(tgt_label) / 50
            # down-sample
            mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
            tgt_label[tgt_label == -1] = 0  # set area B to 0

            # positive sample annealing
            conf = self.cat_data_from_list(batch_list, 'conf')
            labeled_pos = tgt_label == 1
            potential_pos = (conf[..., 1] > (1 - annealing_ratio * 0.5))
            unlabeled_potential_pos = torch.logical_and(potential_pos,
                                                        torch.logical_not(labeled_pos))
            if self.sampling['topk']:
                k = int(labeled_pos.sum().item() * (1 + 30 * annealing_ratio))
                topk = torch.topk(conf[..., 1], k)
                is_topk = torch.zeros_like(labeled_pos)
                is_topk[topk.indices] = 1
                topk_potential_pos = torch.logical_and(is_topk, unlabeled_potential_pos)
                unlabeled_potential_pos = topk_potential_pos

            # set potential positive samples label to ignore
            tgt_label[unlabeled_potential_pos] = -1
        else:
            mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
            # mask = torch.ones_like(tgt_label).bool()
            tgt_label[tgt_label == -1] = 0  # set area B to 0

        # get final tgt
        tgt_pts = tgt_pts[mask]
        tgt_label = tgt_label[mask]

        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # boxes_src = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]]
        # ax = draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=tgt_pts[tgt_pts[:, 0] == 0, 1:].cpu().numpy(),
        #     boxes_gt=boxes_src[boxes_src[:, 0] == 0, 1:],
        #     return_ax=True
        # )
        # pts_ = tgt_pts[tgt_label==1]
        # ax = draw_points_boxes_plt(
        #     points=pts_[pts_[:, 0] == 0, 1:].cpu().numpy(),
        #     points_c='r',
        #     ax=ax,
        #     return_ax=True,
        # )
        # pts_ = tgt_pts[tgt_label==-1]
        # draw_points_boxes_plt(
        #     points=pts_[pts_[:, 0] == 0, 1:].cpu().numpy(),
        #     points_c='orange',
        #     filename='/home/yuan/Downloads/tmp1.png',
        #     ax=ax
        # )

        return tgt_pts, tgt_label, mask

    @torch.no_grad()
    def downsample_tgt_pts(self, tgt_label, max_sam):
        selected = torch.ones_like(tgt_label.bool())
        pos = tgt_label == 1
        if pos.sum() > max_sam:
            mask = torch.rand_like(tgt_label[pos].float()) < max_sam / pos.sum()
            selected[pos] = mask

        neg = tgt_label == 0
        if neg.sum() > max_sam:
            mask = torch.rand_like(tgt_label[neg].float()) < max_sam / neg.sum()
            selected[neg] = mask
        return selected



