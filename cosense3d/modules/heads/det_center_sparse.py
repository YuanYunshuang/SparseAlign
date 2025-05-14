from einops import rearrange

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import linear_last
from cosense3d.utils.misc import multi_apply
from cosense3d.modules.losses import build_loss, pred_to_conf_unc
from cosense3d.modules.utils.me_utils import *
from cosense3d.modules.utils.positional_encoding import ratio2coord


class UnitedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 norm='BN',
                 **kwargs):
        super().__init__()
        n_cls = sum([len(c) for c in class_names_each_head])
        out_channel = n_cls + 1 if one_hot_encoding else n_cls
        self.head = linear_last(in_channel, in_channel, out_channel, use_bias, norm)

    def forward(self, x):
        return [self.head(x)]


class SeparatedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 norm='BN',
                 **kwargs):
        super().__init__()
        self.n_head = len(class_names_each_head)
        for i, cls_names in enumerate(class_names_each_head):
            out_channel = len(cls_names)
            if one_hot_encoding:
                out_channel += 1
            setattr(self, f'head_{i}',
                    linear_last(in_channel, in_channel, out_channel, use_bias, norm))

    def forward(self, x):
        out = []
        for i in range(self.n_head):
            out.append(getattr(self, f'head_{i}')(x))
        return out


class UnitedRegHead(nn.Module):
    def __init__(self,
                 reg_channels,
                 in_channel,
                 combine_channels=True,
                 sigmoid_keys=None,
                 use_bias=False,
                 norm='BN',
                 **kwargs):
        super().__init__()
        self.combine_channels = combine_channels
        self.sigmoid_keys = [] if sigmoid_keys is None else sigmoid_keys
        self.reg_channels = {}
        for c in reg_channels:
            name, channel = c.split(':')
            self.reg_channels[name] = int(channel)

        if combine_channels:
            out_channel = sum(list(self.reg_channels.values()))
            self.head = linear_last(in_channel, in_channel, out_channel, use_bias, norm)
        else:
            for name, channel in self.reg_channels.items():
                setattr(self, f'head_{name}',
                        linear_last(in_channel, in_channel, int(channel), use_bias, norm))

    def forward(self, x):
        out_dict = {}
        if self.combine_channels:
            out_tensor = self.head(x)
            ptr = 0
            for k, v in self.reg_channels.items():
                out = out_tensor[:, ptr:ptr+v]
                if k in self.sigmoid_keys:
                    out = out.sigmoid()
                out_dict[k] = [out] # list compatible with separated head
                ptr += v
        else:
            for k in self.reg_channels.keys():
                out_dict[k] = [getattr(self, f'head_{k}')(x)]
        return out_dict


class DetCenterSparse(BaseModule):
    def __init__(self,
                 data_info,
                 stride,
                 class_names_each_head,
                 shared_conv_channel,
                 cls_head_cfg,
                 reg_head_cfg,
                 reg_channels,
                 cls_assigner,
                 box_assigner,
                 loss_cls,
                 loss_box,
                 center_threshold=0.5,
                 generate_roi_scr=False,
                 sin_enc=False,
                 norm='BN',
                 **kwargs):
        super(DetCenterSparse, self).__init__(**kwargs)
        update_me_essentials(self, data_info, stride)
        self.center_threshold = center_threshold
        self.n_heads = len(class_names_each_head)
        self.class_names_each_head = class_names_each_head
        self.generate_roi_scr = generate_roi_scr
        self.sin_enc = sin_enc
        self.reg_heads = []

        self.cls_head = globals()[cls_head_cfg['name']](
            class_names_each_head,
            shared_conv_channel,
            one_hot_encoding=cls_head_cfg.get('one_hot_encoding', True),
            norm=norm
        )
        self.reg_head = globals()[reg_head_cfg['name']](
            reg_channels,
            shared_conv_channel,
            combine_channels=reg_head_cfg['combine_channels'],
            sigmoid_keys=reg_head_cfg['sigmoid_keys'],
            norm=norm
        )

        self.cls_assigner = plugin.build_plugin_module(cls_assigner)
        self.box_assigner = plugin.build_plugin_module(box_assigner)

        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)

        self.out_dict = {'cls': []}
        for name in self.reg_heads:
            self.out_dict[f'reg_{name}'] = []

        self.temp = 1

    def forward(self, stensor_list, **kwargs):
        self.temp += 1
        B = len(stensor_list)
        coor, feat, centers = self.format_input(stensor_list)
        if centers is not None:
            centers = indices2metric(coor, self.voxel_size)
        cls = self.cls_head(feat)
        reg = self.reg_head(feat)

        out_dict = {
            'ctr': centers,
            'cls': cls,
            'reg': reg,
        }

        if self.generate_roi_scr:
            is_edl = 'edl' in self.loss_cls.name.lower()
            conf = [pred_to_conf_unc(x, self.loss_cls.activation, edl=is_edl)[0] for x in cls]
            conf = torch.stack(conf, dim=0).max(dim=0).values
            if len(conf) == 0:
                print('det_coor', coor.shape)
                print('det_feat', feat.shape)
            if is_edl:
                out_dict['scr'] = conf[:, 1:].max(dim=-1).values
            else:
                out_dict['scr'] = conf.max(dim=-1).values
        if not self.training:
            out_dict['preds'], out_dict['conf'] = self.predictions(out_dict)

        return self.format_output(out_dict, B)

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['ctr'][:, 0] == i
            output_new['ctr'].append(output['ctr'][mask, 1:])
            output_new['cls'].append([h_cls[mask] for h_cls in output['cls']])
            output_new['reg'].append({k:[vi[mask] for vi in v] for k, v in output['reg'].items()})
            if 'conf' in output:
                output_new['conf'].append(output['conf'][mask])
            if 'scr' in output:
                output_new['scr'].append(output['scr'][mask])
            if 'preds' in output:
                mask = output['preds']['idx'][:, 0] == i
                preds = {}
                for k, v in output['preds'].items():
                    if k in ['idx', 'box']:
                        preds[k] = v[mask][:, 1:]
                    else:
                        preds[k] = v[mask]
                output_new['preds'].append(preds)

        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def loss(self, batch_list, gt_boxes, gt_labels, gt_mask=None, **kwargs):
        epoch = kwargs.get('epoch', 0)
        centers = [batch['ctr'] for batch in batch_list]
        pred_cls_list = [torch.stack(batch['cls'], dim=0) for batch in batch_list]
        if 'scr' in batch_list[0]:
            pred_scores = [batch['scr'] for batch in batch_list]
        else:
            pred_scores = [pred_to_conf_unc(x)[0][..., 1:].sum(dim=-1) for x in pred_cls_list]
        if gt_mask is not None:
            for i, m in enumerate(gt_mask):
                gt_boxes[i] = gt_boxes[i][m]
                gt_labels[i] = gt_labels[i][m]
        cls_tgt = multi_apply(self.cls_assigner.assign,
                              centers, gt_boxes, gt_labels, pred_scores, **kwargs)

        # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
        # import os
        # for i in range(2):
        #     ctrs_vis = centers[i].detach().cpu().numpy().T
        #     scrs_vis = pred_cls_list[i][0].sigmoid().detach().cpu().numpy().T
        #     gt_vis = (cls_tgt[i] == 1).squeeze().detach().cpu().numpy()
        #     ax = draw_points_boxes_plt(
        #         pc_range=120,
        #         boxes_pred=gt_boxes[0][:, :7].detach().cpu().numpy(),
        #         return_ax=True
        #     )
        #     ax.scatter(ctrs_vis[0], ctrs_vis[1], c=scrs_vis[0], edgecolors='none', marker='.', vmin=0, vmax=1, cmap='jet')
        #     # ax.scatter(ctrs_vis[0][gt_vis], ctrs_vis[1][gt_vis], c='g', edgecolors='none', marker='.')
        #     plt.savefig(f"{os.environ['HOME']}/Downloads/tmp{i}.jpg")
        #     plt.close()

        cls_tgt = torch.cat(cls_tgt, dim=0)

        n_classes = [len(n) for n in self.class_names_each_head]

        # get reg target
        box_tgt = self.box_assigner.assign(
            self.cat_data_from_list(centers, pad_idx=True),
            self.cat_data_from_list(gt_boxes, pad_idx=True),
            self.cat_data_from_list(gt_labels)
        )

        ptr = 0
        loss_cls = 0
        loss_box = 0
        for h in range(self.n_heads):
            # center loss
            cur_cls_src = torch.cat([x[h] for x in pred_cls_list], dim=0).contiguous()
            cur_cls_tgt = cls_tgt[..., ptr:ptr+n_classes[h]].contiguous() # one hot foreground labels

            cared = (cur_cls_tgt >= 0).any(dim=-1)
            cur_cls_src = cur_cls_src[cared]
            cur_cls_tgt = cur_cls_tgt[cared]
            ptr += n_classes[h]

            # convert one-hot to labels
            cur_labels = torch.zeros_like(cur_cls_tgt[..., 0]).long()
            lbl_inds, cls_inds = torch.where(cur_cls_tgt)
            if 'edl' in self.loss_cls.name.lower():
                cur_labels[lbl_inds] = cls_inds + 1
                cur_num_cls = n_classes[h] + 1
                avg_factor = None if self.cls_assigner.pos_neg_ratio else max((cur_labels > 0).sum(), 1)
            elif 'focal' in self.loss_cls.name.lower():
                cur_num_cls = n_classes[h]
                cur_labels += n_classes[h]
                cur_labels[lbl_inds] = cls_inds
                avg_factor = max(len(cls_inds), 1)
            else:
                raise NotImplementedError

            # focal loss encode the last dim of tgt as background
            # labels = pos_mask.new_full((len(pos_mask),), self.num_classes, dtype=torch.long)
            # labels[pos_mask] = 0

            lcenter = self.loss_cls(
                cur_cls_src,
                cur_labels,
                temp=epoch,
                n_cls_override=cur_num_cls,
                avg_factor=avg_factor
            )
            loss_cls = loss_cls + lcenter

            # reg loss
            ind = box_tgt['idx'][h]
            if ind.shape[1] > 0:
                for reg_name, reg_dim in self.reg_head.reg_channels.items():
                    if reg_dim == 0:
                        continue
                    pred_reg = torch.cat([x['reg'][reg_name][h] for x in batch_list], dim=0)
                    cur_reg_src = rearrange(pred_reg, 'n d ... -> n ... d').contiguous()
                    cur_reg_src = cur_reg_src[box_tgt['valid_mask'][h]]
                    cur_reg_tgt = box_tgt[reg_name][h]  # N, C
                    if reg_name == 'dir' and reg_dim == 1 and self.sin_enc:
                        cur_reg_src, cur_reg_tgt = self.add_sin_difference(cur_reg_src, cur_reg_tgt)
                    cur_loss = self.loss_box(cur_reg_src, cur_reg_tgt)

                    loss_box = loss_box + cur_loss

        loss_dict = {'ctr_loss': loss_cls, 'box_loss': loss_box}
        return loss_dict

    @staticmethod
    def add_sin_difference(dir1, dir2):
        rad_pred_encoding = torch.sin(dir1) * torch.cos(dir2)
        rad_tg_encoding = torch.cos(dir1) * torch.sin(dir2)
        return rad_pred_encoding, rad_tg_encoding

    def predictions(self, preds):
        return self.box_assigner.get_predictions(preds)


class MultiLvlDetCenterSparse(DetCenterSparse):
    def __init__(self, nlvls, sparse, *args, **kwargs):
        super(MultiLvlDetCenterSparse, self).__init__(*args, **kwargs)
        self.nlvls = nlvls
        self.sparse = sparse
        self.lidar_range_cuda = nn.Parameter(torch.tensor(self.lidar_range), requires_grad=False)

    def forward(self, feat_in, **kwargs):
        outs_dec, reference_points, reference_inds = self.format_input(feat_in)

        assert outs_dec.isnan().sum() == 0, "found nan in outs_dec."
        pos_dim = reference_points.shape[-1]
        shape = outs_dec.shape
        centers = ratio2coord(reference_points, self.lidar_range_cuda)

        cls = self.cls_head(outs_dec.view(-1, shape[-1]))
        reg = self.reg_head(outs_dec.view(-1, shape[-1]))

        cls = torch.stack(cls, dim=0).view(self.n_heads, *shape[:-1], -1)  # (nhead, nlvl, nbatch, nsample, ncls)
        reg = {k: torch.stack(v, dim=0).view(self.n_heads, *shape[:-1], -1) for k, v in reg.items()}
        pred_boxes = self.box_assigner.box_coder.decode(
            centers.unsqueeze(0).unsqueeze(0).repeat((self.n_heads, self.nlvls,) + (1,) * len(shape[1:])), reg)

        out_dict = {
            'ctr': centers,
            'cls': cls,
            'reg': reg,
            'pred_boxes': pred_boxes
        }

        out_dict['conf'] = pred_to_conf_unc(cls, self.loss_cls.activation)[0]
        if 'edl' in self.loss_cls.name.lower():
            out_dict['scr'] = out_dict['conf'][..., 1:].max(dim=-1).values
        else:
            out_dict['scr'] = out_dict['conf'].max(dim=-1).values

        if not self.training:
            out_dict['preds'], _ = self.predictions(out_dict)

        return self.format_output(out_dict, len(feat_in), reference_inds)

    def format_input(self, feat_in):
        if self.sparse:
            outs_dec = self.cat_data_from_list(feat_in, 'outs_dec').permute(1, 0, 2)
            reference_points = self.cat_data_from_list(feat_in, 'ref_pts', pad_idx=True)
            reference_inds = reference_points[..., 0]
            reference_points = reference_points[..., 1:]
        else:
            outs_dec = self.stack_data_from_list(feat_in, 'outs_dec').permute(1, 0, 2, 3)
            reference_points = self.stack_data_from_list(feat_in, 'ref_pts')
            reference_inds = None
        return outs_dec, reference_points, reference_inds

    def format_output(self, output, B=None, reference_inds=None):
        outs = []
        for i in range(B):
            if self.sparse:
                m = reference_inds == i
            else:
                m = i
            out = {
                    'cls': output['cls'][:, :, m],
                    'reg': {k: v[:, :, m] for k, v in output['reg'].items()},
                    'ctr': output['ctr'][m],
                    'pred_boxes': output['pred_boxes'][:, :, m],
                }
            if 'scr' in output:
                out['scr'] = output['scr'][:, :, m]
            if 'preds' in output:
                mask = output['preds']['idx'][:, 0] == i
                preds = {}
                for k, v in output['preds'].items():
                    if k in ['idx', 'box']:
                        preds[k] = v[mask][:, 1:]
                    else:
                        preds[k] = v[mask]
                out['preds'] = preds
            outs.append(out)

        return {self.scatter_keys[0]: outs}

    def loss(self, batch_list, gt_boxes, gt_labels, **kwargs):
        epoch = kwargs.get('epoch', 0)
        centers = [batch['ctr'] for batch in batch_list for _ in range(self.nlvls)]
        pred_cls_list = [x for batch in batch_list for x in batch['cls'].transpose(1, 0)]
        pred_scores = [x for batch in batch_list for x in batch['scr'].transpose(1, 0)]

        cls_tgt = multi_apply(self.cls_assigner.assign,
                              centers, gt_boxes, gt_labels, pred_scores, **kwargs)
        cls_tgt = torch.cat(cls_tgt, dim=0)

        n_classes = [len(n) for n in self.class_names_each_head]

        # get reg target
        box_tgt = self.box_assigner.assign(
            self.cat_data_from_list([batch['ctr'] for batch in batch_list], pad_idx=True),
            self.cat_data_from_list(gt_boxes, pad_idx=True),
            self.cat_data_from_list(gt_labels)
        )

        ptr = 0
        loss_cls = 0
        loss_box = 0
        for h in range(self.n_heads):
            # center loss
            cur_cls_src = torch.cat([x[h] for x in pred_cls_list], dim=0).contiguous()
            cur_cls_tgt = cls_tgt[..., ptr:ptr+n_classes[h]].contiguous() # one hot foreground labels

            cared = (cur_cls_tgt >= 0).any(dim=-1)
            cur_cls_src = cur_cls_src[cared]
            cur_cls_tgt = cur_cls_tgt[cared]
            ptr += n_classes[h]

            # convert one-hot to labels
            cur_labels = torch.zeros_like(cur_cls_tgt[..., 0]).long()
            lbl_inds, cls_inds = torch.where(cur_cls_tgt)
            cur_labels[lbl_inds] = cls_inds + 1

            if self.cls_assigner.pos_neg_ratio:
                avg_factor = None
            else:
                avg_factor = max((cur_labels > 0).sum(), 1)
            lcenter = self.loss_cls(
                cur_cls_src,
                cur_labels,
                temp=epoch,
                n_cls_override=n_classes[h] + 1,
                avg_factor=avg_factor
            )
            loss_cls = loss_cls + lcenter

            # reg loss
            ind = box_tgt['idx'][h]
            if ind.shape[1] > 0:
                for reg_name, reg_dim in self.reg_head.reg_channels.items():
                    pred_reg = torch.cat([x['reg'][reg_name][h].view(-1, reg_dim) for x in batch_list], dim=0)
                    cur_reg_src = rearrange(pred_reg, 'n d ... -> n ... d').contiguous()
                    cur_reg_src = cur_reg_src[torch.cat([box_tgt['valid_mask'][h]] * self.nlvls, dim=0)]
                    cur_reg_tgt = torch.cat([box_tgt[reg_name][h]] * self.nlvls, dim=0)  # N, C
                    cur_loss = self.loss_box(cur_reg_src, cur_reg_tgt)

                    loss_box = loss_box + cur_loss

        loss_dict = {'ctr_loss': loss_cls, 'box_loss': loss_box}
        return loss_dict

    def predictions(self, preds):
        return self.box_assigner.get_predictions({
            'ctr': preds['ctr'],
            'cls': preds['cls'][:, -1],
            'reg': {k: v[:, -1] for k, v in preds['reg'].items()}
        })


