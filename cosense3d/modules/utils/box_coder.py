import copy
import math
import torch

from cosense3d.ops.utils import points_in_boxes_gpu


def build_box_coder(type, **kwargs):
    return globals()[type](**kwargs)


class ResidualBoxCoder(object):
    def __init__(self, mode: str='simple_dist'):
        """
        :param mode: str, simple_dist | sin_cos_dist | compass_rose
        """
        self.mode = mode
        if mode == 'simple_dist':
            self.code_size = 7
        elif mode == 'sin_cos_dist':
            self.code_size = 8
        elif mode == 'compass_rose':
            self.code_size = 10
            self.cls_code_size = 2
        else:
            raise NotImplementedError

    def encode_direction(self, ra, rg):
        if self.mode == 'simple_dist':
            reg = (rg - ra).view(-1, 1)
            return reg, None
        elif self.mode == 'sin_cos_dist':
            rgx = torch.cos(rg)
            rgy = torch.sin(rg)
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            rtx = rgx - rax
            rty = rgy - ray
            ret = [rtx, rty]
            reg = torch.stack(ret, dim=-1)  # N 2
            return reg, None
        elif self.mode == 'compass_rose':
            # encode box directions
            rgx = torch.cos(rg).view(-1, 1)  # N 1
            rgy = torch.sin(rg).view(-1, 1)  # N 1
            ra_ext = torch.cat([ra, ra + math.pi], dim=-1)  # N 2, invert
            rax = torch.cos(ra_ext)  # N 2
            ray = torch.sin(ra_ext)  # N 2
            # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
            # we use arccos instead of a-b to control the difference in 0-pi
            diff_angle = torch.arccos(rax * rgx + ray * rgy)  # N 2
            dir_score = 1 - diff_angle / math.pi  # N 2
            rtx = rgx - rax  # N 2
            rty = rgy - ray  # N 2

            dir_score = dir_score  # N 2
            ret = [rtx, rty]
            reg = torch.cat(ret, dim=-1)  # N 4
            return reg, dir_score
        else:
            raise NotImplementedError

    def decode_direction(self, ra, vt, dir_scores=None):
        if self.mode == 'simple_dist':
            rg = vt + ra
            return rg
        elif self.mode == 'sin_cos_dist':
            rax = torch.cos(ra)
            ray = torch.sin(ra)
            va = torch.cat([rax, ray], dim=-1)
            vg = vt + va
            rg = torch.atan2(vg[..., 1], vg[..., 0])
            return rg
        elif self.mode == 'compass_rose':
            ra_ext = torch.cat([ra, ra + math.pi], dim=-1)  # N 2, invert
            rax = torch.cos(ra_ext)  # N 2
            ray = torch.sin(ra_ext)  # N 2
            va = torch.cat([rax, ray], dim=-1)
            vg = vt + va
            rg = torch.atan2(vg[..., 2:], vg[..., :2]).view(-1, 2)

            dirs = torch.argmax(dir_scores, dim=-1).view(-1)
            rg = rg[torch.arange(len(rg)), dirs].view(len(vt), -1, 1)
            return rg
        else:
            raise NotImplementedError

    def encode(self, anchors, boxes):
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

        reg_dir, dir_score = self.encode_direction(ra, rg)
        ret = [xt, yt, zt, lt, wt, ht, reg_dir]
        reg = torch.cat(ret, dim=1)  # N 6+4

        return reg, dir_score

    def decode(self, anchors, boxes_enc, dir_scores=None):
        xa, ya, za, la, wa, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, lt, wt, ht = torch.split(boxes_enc[..., :6], 1, dim=-1)
        vt = boxes_enc[..., 6:]

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha

        rg = self.decode_direction(ra, vt, dir_scores)

        return torch.cat([xg, yg, zg, lg, wg, hg, rg], dim=-1)


class CenterBoxCoder(object):
    def __init__(self, with_velo=False, with_pred=False, reg_radius=1.6, z_offset=1.0):
        self.with_velo = with_velo
        self.with_pred = with_pred
        self.reg_radius = reg_radius
        self.z_offset = z_offset
        self.pred_max_offset = 2.0 + reg_radius

    def encode(self, centers, gt_boxes, meter_per_pixel, gt_preds=None):
        """

        :param centers: (N, 3)
        :param gt_boxes: (N, 8) [batch_idx, x, y, z, l, w, h, r]
        :param meter_per_pixel: tuple with 2 elements
        :param gt_preds:
        :return:
        """
        if isinstance(meter_per_pixel, list):
            assert meter_per_pixel[0] == meter_per_pixel[1], 'only support unified pixel size for x and y'
            # TODO: adapt meter per pixel
            meter_per_pixel = meter_per_pixel[0]
        if len(gt_boxes) == 0:
            valid = torch.zeros_like(centers[:, 0]).bool()
            res = None, None, None, valid
            if self.with_velo:
                res = res + (None,)
            return res

        # match centers and gt_boxes
        dist_ctr_to_box = torch.norm(centers[:, 1:3].unsqueeze(1)
                                     - gt_boxes[:, 1:3].unsqueeze(0), dim=-1)
        cc, bb = torch.meshgrid(centers[:, 0], gt_boxes[:, 0], indexing='ij')
        dist_ctr_to_box[cc != bb] = 1000
        min_dists, box_idx_of_pts = dist_ctr_to_box.min(dim=1)
        diagnal = torch.norm(gt_boxes[:, 4:6].mean(dim=0) / 2)
        valid = min_dists < max(diagnal, meter_per_pixel[0])
        # valid = min_dists < self.reg_radius
        valid_center, valid_box = centers[valid], gt_boxes[box_idx_of_pts[valid]]
        valid_pred = None
        if self.with_pred and gt_preds is not None:
            valid_pred = gt_preds[box_idx_of_pts[valid]]

        xc, yc = torch.split(valid_center[:, 1:3], 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(valid_box[:, 1:8], 1, dim=-1)

        xt = xg - xc
        yt = yg - yc
        zt = zg # + self.z_offset

        lt = torch.log(lg)
        wt = torch.log(wg)
        ht = torch.log(hg)

        # encode box directions
        rgx = torch.cos(rg).view(-1, 1)  # N 1
        rgy = torch.sin(rg).view(-1, 1)  # N 1
        ra = torch.arange(0, 2, 0.5).to(xc.device) * math.pi
        ra_ext = torch.ones_like(valid_box[:, :4]) * ra.view(-1, 4)  # N 4
        rax = torch.cos(ra_ext)  # N 4
        ray = torch.sin(ra_ext)  # N 4
        # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        # we use arccos instead of a-b to control the difference in 0-pi
        diff_angle = torch.arccos(rax * rgx + ray * rgy)  # N 4
        dir_score = 1 - diff_angle / math.pi  # N 4
        rtx = rgx - rax  # N 4
        rty = rgy - ray  # N 4

        reg_box = torch.cat([xt, yt, zt, lt, wt, ht], dim=1)  # N 6
        reg_dir = torch.cat([rtx, rty], dim=1)  # N 8
        # reg_box[..., :3] /= self.reg_radius

        res = (reg_box, reg_dir, dir_score, valid)

        if self.with_velo:
            res = res + (valid_box[:, 8:10],)
        elif valid_box.shape[-1] > 8:
            res = res + (valid_box[:, 8:10],)
        if self.with_pred and valid_pred is not None:
            prev_angles = valid_box[:, 7:8]
            preds_tgt = []
            mask = []
            for i, boxes in enumerate(valid_pred.transpose(1, 0)):
                # some gt_boxes do not have gt successors, zero padded virtual successors are used to align the number
                # of boxes between gt_boxes and gt_preds, when calculate preds loss, these boxes should be ignored.
                mask.append(boxes.any(dim=-1, keepdim=True).float())
                diff_xy = (boxes[:, :2] - valid_center[:, 1:3]) # / self.pred_max_offset
                diff_z = boxes[:, 2:3] # + self.z_offset
                diff_cos = torch.cos(boxes[:, 3:]) - torch.cos(prev_angles)
                diff_sin = torch.sin(boxes[:, 3:]) - torch.sin(prev_angles)
                preds_tgt.append(torch.cat([diff_xy, diff_z, diff_cos, diff_sin], dim=-1) / (i + 2))
            preds_tgt = torch.cat(preds_tgt, dim=-1)
            mask = torch.cat(mask, dim=-1).all(dim=-1, keepdim=True)
            res = res + (torch.cat([mask, preds_tgt], dim=-1),)
        return res

    def decode(self, centers, reg):
        """

        :param centers: Tensor (N, 3) or (B, N, 2+).
        :param reg: dict,
            box - (N, 6) or (B, N, 6)
            dir - (N, 8) or (B, N, 8)
            scr - (N, 4) or (B, N, 4)
            vel - (N, 2) or (B, N, 2), optional
            pred - (N, 5) or (B, N, 5), optional
        :return: decoded bboxes.
        """
        if centers.ndim > 2:
            xc, yc = torch.split(centers[..., 0:2], 1, dim=-1)
        else:
            xc, yc = torch.split(centers[..., 1:3], 1, dim=-1)
        # reg['box'][..., :3] *= self.reg_radius
        xt, yt, zt, lt, wt, ht = torch.split(reg['box'], 1, dim=-1)

        xo = xt + xc
        yo = yt + yc
        zo = zt #- self.z_offset

        lo = torch.exp(lt)
        wo = torch.exp(wt)
        ho = torch.exp(ht)

        # decode box directions
        scr_max, max_idx = reg['scr'].max(dim=-1)
        shape = max_idx.shape
        max_idx = max_idx.view(-1)
        ii = torch.arange(len(max_idx))
        ra = max_idx.float() * 0.5 * math.pi
        ct = reg['dir'][..., :4].view(-1, 4)[ii, max_idx] + torch.cos(ra)
        st = reg['dir'][..., 4:].view(-1, 4)[ii, max_idx] + torch.sin(ra)
        ro = torch.atan2(st.view(*shape), ct.view(*shape)).unsqueeze(-1)

        if centers.ndim > 2:
            # dense tensor
            ret = torch.cat([xo, yo, zo, lo, wo, ho, ro], dim=-1)
        else:
            # sparse tensor with batch indices
            ret = torch.cat([centers[..., :1], xo, yo, zo, lo, wo, ho, ro], dim=-1)

        if self.with_velo:
            ret = torch.cat([ret, reg['vel']], dim=-1)
        if self.with_pred:
            pred = reg['pred'].clone()
            b, n, c = pred.shape
            pred_len = c // 5
            mul = torch.arange(1, pred_len + 1, device=pred.device, dtype=pred.dtype)
            pred = pred.view(b, n, -1, 5) * mul.view(1, 1, -1, 1)
            xy = pred[..., :2] + centers[..., :2].unsqueeze(-2)
            z = pred[..., 2:3]
            r = torch.atan2(pred[..., 4] + st.view(*shape, 1), pred[..., 3] + ct.view(*shape, 1)).unsqueeze(-1)
            lwh = torch.cat([lo, wo, ho], dim=-1).unsqueeze(-2).repeat(1, 1, pred_len, 1)
            pred = torch.cat([xy, z, lwh, r], dim=-1)
            ret = (ret,  pred)

        return ret


class BoxPredCoder(object):
    def __init__(self, with_velo=False):
        self.with_velo = with_velo

    def encode(self, centers, gt_boxes, meter_per_pixel, gt_preds):
        """

        :param centers: (N, 3)
        :param gt_boxes: (N, 8) [batch_idx, x, y, z, l, w, h, r]
        :param meter_per_pixel: tuple with 2 elements
        :param gt_preds: (N, 8) [batch_idx, x, y, z, l, w, h, r], gt boxes to be predicted
        :return: encoded bbox targets.
        """
        if isinstance(meter_per_pixel, list):
            assert meter_per_pixel[0] == meter_per_pixel[1], 'only support unified pixel size for x and y'
            # TODO: adapt meter per pixel
            meter_per_pixel = meter_per_pixel[0]
        if len(gt_boxes) == 0:
            valid = torch.zeros_like(centers[:, 0]).bool()
            res = None, None, None, valid
            if self.with_velo:
                res = res + (None,)
            return res

        # match centers and gt_boxes
        dist_ctr_to_box = torch.norm(centers[:, 1:3].unsqueeze(1)
                                     - gt_boxes[:, 1:3].unsqueeze(0), dim=-1)
        cc, bb = torch.meshgrid(centers[:, 0], gt_boxes[:, 0], indexing='ij')
        dist_ctr_to_box[cc != bb] = 1000
        min_dists, box_idx_of_pts = dist_ctr_to_box.min(dim=1)
        diagnal = torch.norm(gt_boxes[:, 4:6].mean(dim=0) / 2)
        valid = min_dists < max(diagnal, meter_per_pixel[0])
        # valid = min_dists < self.reg_radius
        valid_center = centers[valid]
        valid_box = gt_preds[box_idx_of_pts[valid]]

        xc, yc = torch.split(valid_center[:, 1:3], 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(valid_box[:, 1:8], 1, dim=-1)

        xt = xg - xc
        yt = yg - yc
        zt = zg # + self.z_offset

        lt = torch.log(lg)
        wt = torch.log(wg)
        ht = torch.log(hg)

        # encode box directions
        rgx = torch.cos(rg).view(-1, 1)  # N 1
        rgy = torch.sin(rg).view(-1, 1)  # N 1
        ra = torch.arange(0, 2, 0.5).to(xc.device) * math.pi
        ra_ext = torch.ones_like(valid_box[:, :4]) * ra.view(-1, 4)  # N 4
        rax = torch.cos(ra_ext)  # N 4
        ray = torch.sin(ra_ext)  # N 4
        # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
        # we use arccos instead of a-b to control the difference in 0-pi
        diff_angle = torch.arccos(rax * rgx + ray * rgy)  # N 4
        dir_score = 1 - diff_angle / math.pi  # N 4
        rtx = rgx - rax  # N 4
        rty = rgy - ray  # N 4

        reg_box = torch.cat([xt, yt, zt, lt, wt, ht], dim=1)  # N 6
        reg_dir = torch.cat([rtx, rty], dim=1)  # N 8
        # reg_box[..., :3] /= self.reg_radius

        res = (reg_box, reg_dir, dir_score, valid)

        if self.with_velo:
            res = res + (valid_box[:, 8:10],)
        elif valid_box.shape[-1] > 8:
            res = res + (valid_box[:, 8:10],)
        return res

    def decode(self, centers, reg):
        """

        :param centers: Tensor (N, 3) or (B, N, 2+).
        :param reg: dict,
            box - (N, 6) or (B, N, 6)
            dir - (N, 8) or (B, N, 8)
            scr - (N, 4) or (B, N, 4)
            vel - (N, 2) or (B, N, 2), optional
            pred - (N, 5) or (B, N, 5), optional
        :return: decoded bboxes.
        """
        if centers.ndim > 2:
            xc, yc = torch.split(centers[..., 0:2], 1, dim=-1)
        else:
            xc, yc = torch.split(centers[..., 1:3], 1, dim=-1)
        # reg['box'][..., :3] *= self.reg_radius
        xt, yt, zt, lt, wt, ht = torch.split(reg['box'], 1, dim=-1)

        xo = xt + xc
        yo = yt + yc
        zo = zt #- self.z_offset

        lo = torch.exp(lt)
        wo = torch.exp(wt)
        ho = torch.exp(ht)

        # decode box directions
        scr_max, max_idx = reg['scr'].max(dim=-1)
        shape = max_idx.shape
        max_idx = max_idx.view(-1)
        ii = torch.arange(len(max_idx))
        ra = max_idx.float() * 0.5 * math.pi
        ct = reg['dir'][..., :4].view(-1, 4)[ii, max_idx] + torch.cos(ra)
        st = reg['dir'][..., 4:].view(-1, 4)[ii, max_idx] + torch.sin(ra)
        ro = torch.atan2(st.view(*shape), ct.view(*shape)).unsqueeze(-1)

        if centers.ndim > 2:
            # dense tensor
            ret = torch.cat([xo, yo, zo, lo, wo, ho, ro], dim=-1)
        else:
            # sparse tensor with batch indices
            ret = torch.cat([centers[..., :1], xo, yo, zo, lo, wo, ho, ro], dim=-1)

        if self.with_velo:
            ret = torch.cat([ret, reg['vel']], dim=-1)

        return ret


class CenterBoxCoderV2(object):
    def __init__(self,
                 with_velo=False,
                 reg_radius=1.6,
                 z_offset=1.0,
                 mode='simple'):
        self.with_velo = with_velo
        self.reg_radius = reg_radius
        self.z_offset = z_offset
        self.pred_max_offset = 2.0 + reg_radius
        self.mode = mode

    def encode(self, centers, gt_boxes, meter_per_pixel, gt_preds=None):
        """

        :param centers: (N, 3)
        :param gt_boxes: (N, 8) [batch_idx, x, y, z, l, w, h, r]
        :param meter_per_pixel: tuple with 2 elements
        :param gt_preds:
        :return:
        """
        if isinstance(meter_per_pixel, list):
            assert meter_per_pixel[0] == meter_per_pixel[1], 'only support unified pixel size for x and y'
            # TODO: adapt meter per pixel
            meter_per_pixel = meter_per_pixel[0]
        if len(gt_boxes) == 0:
            valid = torch.zeros_like(centers[:, 0]).bool()
            res = None, None, None, valid
            if self.with_velo:
                res = res + (None,)
            return res

        # match centers and gt_boxes
        dist_ctr_to_box = torch.norm(centers[:, 1:3].unsqueeze(1)
                                     - gt_boxes[:, 1:3].unsqueeze(0), dim=-1)
        cc, bb = torch.meshgrid(centers[:, 0], gt_boxes[:, 0], indexing='ij')
        dist_ctr_to_box[cc != bb] = 1000
        min_dists, box_idx_of_pts = dist_ctr_to_box.min(dim=1)
        diagnal = torch.norm(gt_boxes[:, 4:6].mean(dim=0) / 2)
        valid = min_dists < max(diagnal, meter_per_pixel[0])
        # valid = min_dists < self.reg_radius
        valid_center, valid_box = centers[valid], gt_boxes[box_idx_of_pts[valid]]

        xc, yc = torch.split(valid_center[:, 1:3], 1, dim=-1)
        xg, yg, zg, lg, wg, hg, rg = torch.split(valid_box[:, 1:8], 1, dim=-1)

        xt = xg - xc
        yt = yg - yc
        zt = zg # + self.z_offset

        lt = torch.log(lg)
        wt = torch.log(wg)
        ht = torch.log(hg)

        reg_box = torch.cat([xt, yt, zt, lt, wt, ht], dim=1)  # N 6

        # encode box directions
        reg_dir, dir_score = self.encode_direction(rg, valid_box, xc.device)


        res = (reg_box, reg_dir, dir_score, valid)

        if self.with_velo:
            res = res + (valid_box[:, 8:10],)
        elif valid_box.shape[-1] > 8:
            res = res + (valid_box[:, 8:10],)
        return res

    def decode(self, centers, reg):
        """

        :param centers: Tensor (N, 3) or (B, N, 2+).
        :param reg: dict,
            box - (N, 6) or (B, N, 6)
            dir - (N, 8) or (B, N, 8)
            scr - (N, 4) or (B, N, 4)
            vel - (N, 2) or (B, N, 2), optional
            pred - (N, 5) or (B, N, 5), optional
        :return: decoded bboxes.
        """
        if centers.ndim > 2:
            xc, yc = torch.split(centers[..., 0:2], 1, dim=-1)
        else:
            xc, yc = torch.split(centers[..., 1:3], 1, dim=-1)
        # reg['box'][..., :3] *= self.reg_radius
        xt, yt, zt, lt, wt, ht = torch.split(reg['box'], 1, dim=-1)

        xo = xt + xc
        yo = yt + yc
        zo = zt #- self.z_offset

        lo = torch.exp(lt)
        wo = torch.exp(wt)
        ho = torch.exp(ht)

        # decode box directions
        ro = self.decode_direction(reg)

        if centers.ndim > 2:
            # dense tensor
            ret = torch.cat([xo, yo, zo, lo, wo, ho, ro], dim=-1)
        else:
            # sparse tensor with batch indices
            ret = torch.cat([centers[..., :1], xo, yo, zo, lo, wo, ho, ro], dim=-1)

        if self.with_velo:
            ret = torch.cat([ret, reg['vel']], dim=-1)

        return ret

    def encode_direction(self, rg, valid_box, device):
        if self.mode == 'simple':
            reg = rg.view(-1, 1)
            return reg, None
        elif self.mode == 'sin_cos':
            rgx = torch.cos(rg).view(-1, 1)
            rgy = torch.sin(rg).view(-1, 1)
            # ra = torch.arange(0, 1, 0.5).to(device) * math.pi
            # rax = torch.cos(ra)
            # ray = torch.sin(ra)
            # rtx = rgx - rax
            # rty = rgy - ray
            # ret = [rtx, rty]
            ret = [rgx, rgy]
            reg = torch.cat(ret, dim=-1)  # N 2
            return reg, None
        elif self.mode == 'compass_rose':
            rgx = torch.cos(rg).view(-1, 1)  # N 1
            rgy = torch.sin(rg).view(-1, 1)  # N 1
            ra = torch.arange(0, 2, 0.5).to(device) * math.pi
            ra_ext = torch.ones_like(valid_box[:, :4]) * ra.view(-1, 4)  # N 4
            rax = torch.cos(ra_ext)  # N 4
            ray = torch.sin(ra_ext)  # N 4
            # cos(a - b) = cos(a)cos(b) + sin(a)sin(b)
            # we use arccos instead of a-b to control the difference in 0-pi
            diff_angle = torch.arccos(rax * rgx + ray * rgy)  # N 4
            dir_score = 1 - diff_angle / math.pi  # N 4
            rtx = rgx - rax  # N 4
            rty = rgy - ray  # N 4

            reg_dir = torch.cat([rtx, rty], dim=1)  # N 8
            # reg_box[..., :3] /= self.reg_radius
            return reg_dir, dir_score
        else:
            raise NotImplementedError

    def decode_direction(self, reg):
        if self.mode == 'simple':
            rg = reg['dir']
            return rg
        elif self.mode == 'sin_cos':
            # ra = torch.arange(0, 1, 0.5).to(reg['dir'].device) * math.pi
            # rax = torch.cos(ra)
            # ray = torch.sin(ra)
            # va = torch.cat([rax, ray], dim=-1).view(1, 1, 4)
            # vg = reg['dir'] + va
            vg = reg['dir']
            rg = torch.atan2(vg[..., 1], vg[..., 0]).unsqueeze(-1)
            return rg
        elif self.mode == 'compass_rose':
            scr_max, max_idx = reg['scr'].max(dim=-1)
            shape = max_idx.shape
            max_idx = max_idx.view(-1)
            ii = torch.arange(len(max_idx))
            ra = max_idx.float() * 0.5 * math.pi
            ct = reg['dir'][..., :4].view(-1, 4)[ii, max_idx] + torch.cos(ra)
            st = reg['dir'][..., 4:].view(-1, 4)[ii, max_idx] + torch.sin(ra)
            ro = torch.atan2(st.view(*shape), ct.view(*shape)).unsqueeze(-1)
            return ro
        else:
            raise NotImplementedError
