import os

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.common import pad_r, linear_last, cat_coor_with_idx
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.modules.losses import edl, build_loss
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.plugin.attn import NeighborhoodAttention


class BEV(BaseModule):
    def __init__(self,
                 data_info,
                 in_dim,
                 stride,
                 target_assigner,
                 loss_cls,
                 num_cls=1,
                 class_names_each_head=None,
                 down_sample_tgt=False,
                 generate_roi_scr=True,
                 **kwargs):
        super(BEV, self).__init__(**kwargs)
        self.in_dim = in_dim
        self.class_names_each_head = class_names_each_head
        self.down_sample_tgt = down_sample_tgt
        self.stride = stride
        self.num_cls = num_cls
        self.generate_roi_scr = generate_roi_scr
        for k, v in data_info.items():
            setattr(self, k, v)
        update_me_essentials(self, data_info, self.stride)

        self.reg_layer = linear_last(in_dim, 32, num_cls, bias=True)

        self.tgt_assigner = build_plugin_module(target_assigner)
        self.loss_cls = build_loss(**loss_cls)
        self.is_edl = True if 'edl' in self.loss_cls.name.lower() else False

    def forward(self, stensor_list, **kwargs):
        coor, feat, ctr = self.format_input(stensor_list)

        if self.training and self.down_sample_tgt:
            coor, feat = self.down_sample(coor, feat)

        centers = indices2metric(coor, self.voxel_size)
        reg = self.reg_layer(feat)

        conf, unc = self.tgt_assigner.get_predictions(
            reg, self.is_edl, getattr(self.loss_cls, 'activation'))

        out = {
            'ctr': centers,
            'reg': reg,
            'conf': conf,
            'unc': unc,
        }
        if self.generate_roi_scr:
            out['scr'] = conf.max(dim=-1).values

        return self.format_output(out, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['ctr'][:, 0] == i
            output_new['ctr'].append(output['ctr'][mask, 1:])
            output_new['reg'].append(output['reg'][mask])
            output_new['conf'].append(output['conf'][mask])
            output_new['unc'].append(output['unc'][mask])
            if 'scr' in output_new:
                output_new['scr'].append(output['scr'][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

    def loss(self, batch_list, gt_boxes, gt_labels, **kwargs):
        tgt_pts = self.cat_data_from_list(batch_list, 'ctr', pad_idx=True)
        boxes_vis = gt_boxes[0][:, :7].detach().cpu().numpy()
        gt_boxes = self.cat_data_from_list(gt_boxes, pad_idx=True)
        conf = self.cat_data_from_list(batch_list, 'conf')
        tgt_pts, tgt_label, valid = self.tgt_assigner.assign(
            tgt_pts, gt_boxes[:, :8], len(batch_list), conf, **kwargs)
        epoch_num = kwargs.get('epoch', 0)
        reg = self.cat_data_from_list(batch_list, 'reg')

        # if kwargs['itr'] % 100 == 0:
        #     from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        #     from matplotlib import colormaps
        #     jet = colormaps['jet']
        #     points = batch_list[0]['ctr'].detach().cpu().numpy()
        #     scores = batch_list[0]['conf'][:, self.num_cls - 1:].detach().cpu().numpy()
        #     ax = draw_points_boxes_plt(
        #         pc_range=[-144, -41.6, -3.0, 144, 41.6, 1.0],
        #         # points=points,
        #         boxes_gt=boxes_vis,
        #         return_ax=True
        #     )
        #     ax.scatter(points[:, 0], points[:, 1], c=scores, cmap=jet, s=3, marker='s', vmin=0, vmax=1)
        #     plt.savefig(f"{os.environ['HOME']}/Downloads/tmp1.jpg")
        #     plt.close()

        if valid is None:
            # targets are not down-sampled
            avg_factor = max(tgt_label.sum(), 1)
            loss_cls = self.loss_cls(
                reg,
                tgt_label,
                temp=epoch_num,
                avg_factor=avg_factor
            )
        else:
            # negative targets are not down-sampled to a ratio to the positive samples
            loss_cls = self.loss_cls(
                reg[valid],
                tgt_label,
                temp=epoch_num,
            )
        loss_dict = {'bev_loss': loss_cls}
        return loss_dict


class BEVMultiResolution(BaseModule):
    def __init__(self, strides, strides_for_loss, **kwargs):
        super().__init__(**kwargs)
        self.strides = strides
        self.strides_for_loss = strides_for_loss
        for s in strides:
            kwargs['stride'] = s
            setattr(self, f'head_p{s}', BEV(**kwargs))

    def forward(self, stensor_list, *args, **kwargs):
        out_list = [{} for b in range(len(stensor_list))]
        for s in self.strides:
            out = getattr(self, f'head_p{s}')(stensor_list)[self.scatter_keys[0]]
            for i, x in enumerate(out):
                out_list[i][f'p{s}'] = x

        return {self.scatter_keys[0]: out_list}

    def loss(self, batch_list, gt_boxes, gt_labels, **kwargs):
        loss_dict = {}
        for s in self.strides_for_loss:
            ldict = getattr(self, f'head_p{s}').loss(
                [l[f'p{s}'] for l in batch_list], gt_boxes, gt_labels, **kwargs)
            for k, v in ldict.items():
                loss_dict[f'{k}_s{s}'] = v
        return loss_dict


class ContinuousBEV(BaseModule):
    def __init__(self,
                 out_channels,
                 data_info,
                 in_dim,
                 stride,
                 context_decoder,
                 target_assigner,
                 loss_cls,
                 class_names_each_head=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.class_names_each_head = class_names_each_head
        self.stride = stride
        for k, v in data_info.items():
            setattr(self, k, v)
        update_me_essentials(self, data_info, self.stride)

        self.context_decoder = build_plugin_module(context_decoder)

        self.reg_layer = linear_last(in_dim, 32, out_channels, bias=True)

        self.tgt_assigner = build_plugin_module(target_assigner)
        self.loss_cls = build_loss(**loss_cls)

    @torch.no_grad()
    def sample_reference_points(self, centers, gt_boxes, gt_labels):
        gt_boxes = self.cat_data_from_list(gt_boxes, pad_idx=True)
        if self.training:
            new_pts = centers.clone()
            new_pts[:, 1:] += (torch.rand_like(centers[:, 1:]) - 0.5) * self.res[0]
            ref_pts, ref_label, _ = self.tgt_assigner.assign(
                new_pts, gt_boxes, len(gt_boxes))
        else:
            ref_pts, ref_label, _ = self.tgt_assigner.assign(
                centers, gt_boxes, len(gt_boxes), down_sample=False)
        return ref_pts, ref_label

    def get_evidence(self, ref_pts, coor, feat):
        raise NotImplementedError

    def forward(self, stensor_list, gt_boxes, gt_labels, **kwargs):
        coor, feat, ctr = self.format_input(stensor_list)
        centers = indices2metric(coor, self.voxel_size)
        ref_pts, ref_label = self.sample_reference_points(
            centers, gt_boxes, gt_labels)
        evidence = self.get_evidence(ref_pts, coor, feat)
        conf, unc = edl.evidence_to_conf_unc(evidence)

        out = {
            'ref_pts': ref_pts,
            'ref_lbls': ref_label,
            'evi': evidence,
            'conf': conf,
            'unc': unc
        }

        return self.format_output(out, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['ref_pts'][:, 0] == i
            output_new['ref_pts'].append(output['ref_pts'][mask, 1:])
            output_new['ref_lbls'].append(output['ref_lbls'][mask])
            output_new['evi'].append(output['evi'][mask])
            output_new['conf'].append(output['conf'][mask])
            output_new['unc'].append(output['unc'][mask])
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

    def loss(self, batch_list, **kwargs):
        tgt_lbl = self.cat_data_from_list(batch_list, 'ref_lbls')
        epoch_num = kwargs.get('epoch', 0)
        evidence = self.cat_data_from_list(batch_list, 'evi')
        # avg_factor = max(tgt_label.sum(), 1)
        loss_cls = self.loss_cls(
            evidence,
            tgt_lbl,
            temp=epoch_num,
            # avg_factor=avg_factor
        )
        loss_dict = {'bev_loss': loss_cls}
        return loss_dict


class ContiGevBEV(ContinuousBEV):

    def get_evidence(self, ref_pts, coor, feat):
        reg = self.reg_layer(feat)
        reg = self.context_decoder(ref_pts, coor, reg)
        return reg


class ContiAttnBEV(ContinuousBEV):

    def get_evidence(self, ref_pts, coor, feat):
        ref_context = self.context_decoder(ref_pts, coor, feat)
        reg = self.reg_layer(ref_context)
        return reg.relu()





