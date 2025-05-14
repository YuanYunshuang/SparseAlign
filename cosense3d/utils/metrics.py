import os, logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from cosense3d.ops.iou3d_nms_utils import boxes_iou3d_gpu, boxes_iou_bev
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.utils.misc import save_json, update_dict
from cosense3d.utils.box_utils import mask_boxes_outside_range_torch


class Metric:
    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir

    def add_samples(self, data_dict):
        raise NotImplementedError

    def save_detections(self, filename):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError


class MetricObjDet(Metric):
    def __init__(self, cfg, log_dir, logger, bev=False):
        super(MetricObjDet, self).__init__(cfg, log_dir)
        self.eval_func = cfg['eval_func']
        self.lidar_range = cfg.get('lidar_range', None)
        self.score_metric = cfg.get('score_metric', 'scr')
        self.score_thr = cfg.get('score_thr', 0.0)
        self.logger = logger
        self.samples = []
        self.pred_boxes = {}
        self.gt_boxes = {}
        self.confidences = {}
        self.v_ids = {}
        self.bev = bev
        self.iou_fn = boxes_iou_bev if self.bev else boxes_iou3d_gpu
        self.file_test = os.path.join(log_dir, 'pred.json')
        self.has_test_detections = False
        self.result = {}

    def add_sample(self, name, pred_boxes, gt_boxes, confidences, ids=None):
        self.samples.append(name)
        valid = confidences > self.score_thr
        if self.lidar_range is not None:
            in_range_gt = mask_boxes_outside_range_torch(gt_boxes, self.lidar_range),
            in_range_pred = mask_boxes_outside_range_torch(pred_boxes, self.lidar_range)
            valid = torch.logical_and(valid, in_range_pred)
            gt_boxes = gt_boxes[in_range_gt]
        self.pred_boxes[name] = pred_boxes[valid]
        self.gt_boxes[name] = gt_boxes
        self.confidences[name] = confidences[valid]
        if ids is not None:
            self.v_ids[name] = ids
        ss = name.split("/")
        scenario = ss[0]
        frame = ss[1]
        pred_boxes_np = pred_boxes[valid].cpu().numpy()
        bbx_out = np.zeros((len(pred_boxes_np), 11))
        bbx_out[:, [2, 3, 4, 5, 6, 7, 10]] = pred_boxes_np
        bbx_out[:, 0] = -1  # box id not set
        conf_out = confidences[valid].cpu().numpy()
        if '.' in frame:
            frame, agent_id = frame.split('.')
            fdict = {'agents': {
                agent_id: {
                    'gt_boxes': bbx_out.tolist(),
                    'box_confidences': conf_out.tolist()
                }
            }}
        else:
            fdict = {'meta': {'bbx_center_global': bbx_out.tolist()}}
        update_dict(
            self.result,
            {scenario: {frame: fdict}}
        )

    @torch.no_grad()
    def add_samples(self, out_dict):
        data_dict = out_dict['detections']
        names = data_dict['name']
        for i in range(len(names)):
            self.add_sample(names[i],
                            data_dict['pred_boxes'][i]['box'].float(),
                            data_dict['gt_boxes'][i].float(),
                            data_dict['pred_boxes'][i][self.score_metric])

    def save_detections(self, filename):
        dict_detections = {
            'samples': self.samples,
            'pred_boxes': self.pred_boxes,
            'gt_boxes': self.gt_boxes,
            'confidences': self.confidences,
            'ids': self.v_ids
        }
        torch.save(dict_detections, filename)
        self.has_test_detections = True

    def cal_precision_recall(self, IoU_thr=0.5):
        list_sample = []
        list_confidence = []
        list_tp = []
        N_gt = 0

        for sample in self.samples:
            if len(self.pred_boxes[sample])>0 and len(self.gt_boxes[sample])>0:
                ious = self.iou_fn(self.pred_boxes[sample], self.gt_boxes[sample])
                n, m = ious.shape
                list_sample.extend([sample] * n)
                list_confidence.extend(self.confidences[sample])
                N_gt += len(self.gt_boxes[sample])
                max_iou_pred_to_gts = ious.max(dim=1)
                max_iou_gt_to_preds = ious.max(dim=0)
                tp = max_iou_pred_to_gts[0] > IoU_thr
                is_best_match = max_iou_gt_to_preds[1][max_iou_pred_to_gts[1]] \
                                ==torch.tensor([i for i in range(len(tp))], device=tp.device)
                tp[torch.logical_not(is_best_match)] = False
                list_tp.extend(tp)
            elif len(self.pred_boxes[sample])==0:
                N_gt += len(self.gt_boxes[sample])
            elif len(self.gt_boxes[sample])==0:
                tp = torch.zeros(len(self.pred_boxes[sample]), device=self.pred_boxes[sample].device)
                list_tp.extend(tp.bool())
        order_inds = torch.tensor(list_confidence).argsort(descending=True)
        tp_all = torch.tensor(list_tp)[order_inds]
        list_accTP = tp_all.cumsum(dim=0)
        # list_accFP = torch.logical_not(tp_all).cumsum(dim=0)
        list_precision = list_accTP.float() / torch.arange(1, len(list_sample) + 1)
        list_recall = list_accTP.float() / N_gt
        # plt.plot(list_recall.numpy(), list_precision.numpy(), 'k.')
        # plt.savefig(str(model.run_path / 'auc_thr{}_ncoop{}.png'
        #                 .format(model.cfg['score_threshold'], model.n_coop)))
        # plt.close()

        return list_precision, list_recall

    def cal_ap_all_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L292
        '''

        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    def cal_ap_11_point(self, IoU_thr=0.5):
        '''
        source: https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/7c0bd0489e3fd4ae71fc0bc8f2a67dbab5dbdc9c/lib/Evaluator.py#L315
        '''
        # 11-point interpolated average precision
        prec, rec = self.cal_precision_recall(IoU_thr=IoU_thr)
        mrec = []
        # mrec.append(0)
        [mrec.append(e.item()) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e.item()) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than det_r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above det_r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above det_r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    def summary(self):
        thrs = [0.3, 0.5, 0.7]
        ss = []
        for thr in thrs:
            ap = getattr(self, self.eval_func)(thr)[0]
            ss.append(f"AP@{thr}: {ap:.4f}")
        ss = (f"Score metric: {self.score_metric}\n "
              f"Score thr: {self.score_thr:.2f}\n"
              f"--------------\n"
              + "\n".join(ss) + "\n")
        print(ss)
        self.logger.write(ss)

        os.makedirs(os.path.join(self.log_dir, "jsons"), exist_ok=True)
        for s, sdict in self.result.items():
            save_json(sdict, os.path.join(self.log_dir, "jsons", f'{s}.json'))


class MetricSemSeg(Metric):
    def __init__(self, cfg, run_path, name='test'):
        super(MetricSemSeg, self).__init__(cfg, run_path)
        self.filename = os.path.join(run_path, name)
        self.n_cls = cfg['n_cls']
        # model.result = {
        #     'tp': [],
        #     'tn': [],
        #     'fp': [],
        #     'fn': [],
        #     'N': 0
        # }
        self.result = {
            'area_intersect': torch.zeros(self.n_cls),
            'area_label': torch.zeros(self.n_cls),
            'area_pred': torch.zeros(self.n_cls),
            'area_union': torch.zeros(self.n_cls)
        }

    def add_samples(self, data_dict):
        preds = torch.argmax(data_dict['pred_cls'], dim=1).view(-1, 1)
        tgts = data_dict['tgt_cls'].view(-1, 1)
        # mask = (tgts != 0)
        # preds = preds[mask]
        # tgts = tgts[mask]
        classes = torch.arange(self.n_cls, dtype=preds.dtype, device=preds.device).view(1, -1)
        intersect = preds[preds == tgts]
        area_intersect = (intersect.view(-1, 1) == (classes)).sum(0)
        area_pred = (preds.view(-1, 1) == (classes)).sum(0)
        area_label = (tgts.view(-1, 1) == (classes)).sum(0)
        area_union = area_label + area_label - area_intersect
        self.result['area_intersect'] = self.result['area_intersect'] + area_intersect.cpu()
        self.result['area_label'] = self.result['area_label'] + area_label.cpu()
        self.result['area_pred'] = self.result['area_pred'] + area_pred.cpu()
        self.result['area_union'] = self.result['area_union'] + area_union.cpu()
        # pred_pos = preds.int() == classes
        # pred_neg = torch.logical_not(pred_pos)
        # tgt_pos = tgts.int() == classes
        # tgt_neg = torch.logical_not(tgt_pos)
        # tp = torch.logical_and(pred_pos, tgt_pos).sum(0)
        # tn = torch.logical_and(pred_neg, tgt_neg).sum(0)
        # fp = torch.logical_and(pred_pos, tgt_neg).sum(0)
        # fn = torch.logical_and(pred_neg, tgt_pos).sum(0)
        # acc_ = tp.sum() / len(tgts)
        # model.result['tp'].append(tp)
        # model.result['tn'].append(tn)
        # model.result['fp'].append(fp)
        # model.result['fn'].append(fn)
        # model.result['N'] += len(tgts)

    def cal_ious_and_accs(self):
        area_intersect = self.result['area_intersect'].sum(0)
        area_label = self.result['area_label'].sum(0)
        area_union = self.result['area_union'].sum(0)
        all_acc = area_intersect.sum() / area_label.sum()
        acc = area_intersect / area_label
        iou = area_intersect / area_union

        result = {
            'all_acc': all_acc,
            'acc': acc,
            'iou': iou
        }
        for k, v in result.items():
            print(k, v)
        return result

    def save_detections(self, filename):
        torch.save(self.result, filename)


class MetricBev(Metric):
    def __init__(self, cfg, run_path, logger, name='test'):
        super(MetricBev, self).__init__(cfg, run_path)
        self.filename = os.path.join(run_path, name)
        self.filename_prefix = ''
        self.logger = logger
        self.cfg = cfg
        self.thrs = torch.arange(0.1, 1.1, 0.1)
        self.iou_sum = 0
        self.iou_cnt = 0
        self.result = {}

    def add_samples(self, out_dict):
        """
        Args:
            out_dict:
                bev:
                    conf: Tensor, (B, H, W, C) or (N, C)
                    unc: Tensor (optional), (B, H, W, C) or (N, C)
                    gt: Tensor, (B, H, W, C) or (N, C)
        """
        self.iou(**out_dict['bev'])

    def iou(self, conf, gt, unc=None):
        """
        Compare the thresholded pred BEV map with the full gt BEV map (including non
        observable area)
        """
        if unc is None:
            pred = conf[..., 1] > 0.5
            mi = torch.logical_and(pred, gt).sum()
            mu = torch.logical_or(pred, gt).sum()
            self.iou_sum += mi / mu
            self.iou_cnt += 1
        else:
            pos_mask = conf[..., 1] > 0.5
            pos_mask = torch.logical_and(pos_mask, unc < 1.0)
            mi = torch.logical_and(pos_mask, gt).sum()
            mu = torch.logical_or(pos_mask, gt).sum()

        self.iou_sum += mi.item() / mu.item()
        self.iou_cnt += 1

        # import matplotlib.pyplot as plt
        # plt.imshow(conf[0, ..., 1].cpu().numpy())
        # plt.show()
        # plt.close()
        # plt.imshow(gt[0].cpu().numpy())
        # plt.show()
        # plt.close()

    def summary(self):
        iou_mean = self.iou_sum / self.iou_cnt * 100

        self.summary_hook()

        self.result = {
            'BEV.iou': iou_mean
        }
        ss = self.format_str(self.result)
        print(ss)
        self.logger.write(ss)

    def summary_hook(self):
        pass

    def format_str(self, result_dict):
        ss = "==================================================================================\n"
        for k, vs in result_dict.items():
            s1 = f"{k:20s} : "
            if isinstance(vs, float):
                s2 = f"{vs:4.1f} \n"
            else:
                s2 = "  ".join([f"{v:4.1f} " for v in vs]) + "\n"
            ss += s1 + s2
        return ss



class MetricMOT(Metric):
    def __init__(self, cfg, log_dir):
        super().__init__(cfg, log_dir)

    def add_samples(self, data_dict):
        pass




