import os

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.common import pad_r, linear_last, cat_coor_with_idx
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses import edl, build_loss
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.plugin.attn import NeighborhoodAttention


class BEVRoadLine(BaseModule):
    def __init__(self,
                 data_info,
                 in_dim,
                 stride,
                 target_assigner,
                 loss_cls,
                 num_cls=1,
                 **kwargs):
        super(BEVRoadLine, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.stride = stride
        self.num_cls = num_cls

        setattr(self, f'p{stride}_cls', linear_last(in_dim, 32, num_cls, bias=True))

        self.tgt_assigner = build_plugin_module(target_assigner)
        self.loss_cls = build_loss(**loss_cls)
        self.is_edl = True if 'edl' in self.loss_cls.name.lower() else False

    def forward(self, stensor_list, **kwargs):
        coor, feat, ctr = self.format_input(stensor_list)
        cls = getattr(self, f'p{self.stride}_cls')(feat)

        out = {
            'coor': coor,
            'ctr': ctr,
            'cls': cls,
        }

        # import matplotlib.pyplot as plt
        # bmsk = coor[:, 0] == 0
        # pts_vis = coor[bmsk]
        # pts_vis = pts_vis[:, 1:].detach().cpu().numpy()
        # scr_vis = cls[bmsk].detach().cpu().numpy().squeeze()
        #
        # fig = plt.figure(figsize=(10, 5))
        # ax = fig.add_subplot()
        # ax.scatter(pts_vis[:, 0], pts_vis[:, 1], c=scr_vis, marker='.', vmin=0, vmax=1, s=3)
        # plt.show()
        # plt.close()

        return self.format_output(out, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['coor'][:, 0] == i
            output_new['coor'].append(output['coor'][mask, 1:])
            output_new['ctr'].append(output['ctr'][mask])
            output_new['cls'].append(output['cls'][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

    def loss(self, batch_list, tgt_pts, **kwargs):
        coor = self.cat_data_from_list(batch_list, 'coor', pad_idx=True)
        tgt_pts = self.cat_data_from_list(tgt_pts, pad_idx=True)
        tgt_label, valid = self.tgt_assigner.assign(
            coor, tgt_pts, len(batch_list), **kwargs)
        epoch_num = kwargs.get('epoch', 0)
        cls = self.cat_data_from_list(batch_list, 'cls')

        # import matplotlib.pyplot as plt
        # bmsk = coor[:, 0] == 0
        # pts_vis = coor[bmsk][valid[bmsk]]
        # pts_vis = pts_vis[:, 1:].detach().cpu().numpy()
        # lbl_vis = tgt_label[bmsk[valid]].detach().cpu().numpy()
        # scr_vis = cls[bmsk][valid[bmsk]].detach().cpu().numpy().squeeze()
        #
        # fig = plt.figure(figsize=(10, 5))
        # axs = fig.subplots(1, 2)
        # axs[0].scatter(pts_vis[:, 0], pts_vis[:, 1], c=lbl_vis, marker='.', vmin=0, vmax=1, s=1)
        # axs[1].scatter(pts_vis[:, 0], pts_vis[:, 1], c=scr_vis, marker='.', vmin=0, vmax=1, s=1)
        # plt.show()
        # plt.close()

        # targets are not down-sampled
        cared = tgt_label >= 0
        n_cared = cared.sum()
        if n_cared == len(tgt_label):
            avg_factor = max(tgt_label.bool().sum(), 1)
        else:
            avg_factor = n_cared
        loss_cls = self.loss_cls(
            cls[valid][cared],
            tgt_label[cared],
            temp=epoch_num,
            avg_factor=avg_factor
        )
        loss_dict = {'rl_loss': loss_cls}
        return loss_dict








