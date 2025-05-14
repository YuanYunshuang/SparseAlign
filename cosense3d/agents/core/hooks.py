
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from importlib import import_module

from cosense3d.ops.utils import points_in_boxes_gpu


class Hooks:
    def __init__(self, cfg):
        self.hooks = []
        if cfg is None:
            return
        for hook_cfg in cfg:
            self.hooks.append(
                globals()[hook_cfg['type']](**hook_cfg)
            )

    def __call__(self, runner, hook_stage, **kwargs):
        for hook in self.hooks:
            getattr(hook, hook_stage)(runner, **kwargs)

    def set_logger(self, logger):
        for hook in self.hooks:
            hook.set_logger(logger)


class BaseHook:
    def __init__(self, **kwargs):
        pass

    def pre_iter(self, runner, **kwargs):
        pass

    def post_iter(self, runner, **kwargs):
        pass

    def pre_epoch(self, runner, **kwargs):
        pass

    def post_epoch(self, runner, **kwargs):
        pass

    def set_logger(self, logger):
        self.logger = logger


class MemoryUsageHook(BaseHook):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def post_iter(self, runner, **kwargs):
        memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        torch.cuda.empty_cache()
        runner.logger.update(memory=memory)


class CPMStatisticHook(BaseHook):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'detection_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_epoch(self, runner, **kwargs):
        cpm_rec = runner.controller.cav_manager.cpm_size_recorder
        thr = runner.controller.cav_manager.cavs[0][0].share_score_thr
        ss = (f"########## CPM size @ {thr} ###########\n"
              f"Mean: {cpm_rec.mean[0] * 4 / 1024:.2f} KB, Std: {cpm_rec.std[0] * 4 / 1024:.2f} KB\n")
        print(ss)
        self.logger.log(ss)


class TrainTimerHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elapsed_time = 0
        self.start_time = None
        self.mean_time_per_itr = None
        self.observations = 0

    def pre_epoch(self, runner, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
            self.last_time = time.time()

    def post_iter(self, runner, **kwargs):
        cur_time = time.time()
        self.elapsed_time = (cur_time - self.start_time) / 3600
        # total_run_iter = (runner.total_iter * (runner.epoch - runner.start_epoch)) + runner.iter
        # time_per_iter = self.elapsed_time / total_run_iter
        time_per_iter = (cur_time - self.last_time) / 3600
        m = self.observations
        if self.mean_time_per_itr is None:
            self.mean_time_per_itr = time_per_iter
        else:
            self.mean_time_per_itr = m / (m + 1) * self.mean_time_per_itr + 1 / (m + 1) * time_per_iter
        iter_remain = runner.total_iter * (runner.total_epochs - runner.epoch + 1) - runner.iter
        time_remain = self.mean_time_per_itr * iter_remain
        runner.logger.update(t_remain=time_remain, t_used=self.elapsed_time)
        self.last_time = cur_time
        self.observations += 1


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def post_epoch(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        self.save(runner, f'epoch{runner.epoch}.pth')
        if runner.epoch > self.max_ckpt:
            if (self.epoch_every is None or not
            (runner.epoch - self.max_ckpt) % self.epoch_every == 0):
                filename = os.path.join(
                    runner.logger.logdir,
                    f'epoch{runner.epoch - self.max_ckpt}.pth')
                if os.path.exists(filename):
                    os.remove(filename)

    def post_iter(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        if self.iter_every is not None and runner.iter % self.iter_every == 0:
            self.save(runner, f'latest.pth')

    @staticmethod
    def save(runner, name):
        save_path = os.path.join(runner.logger.logdir, name)
        print(f'Saving checkpoint to {save_path}.')
        torch.save({
            'epoch': runner.epoch,
            'model': runner.forward_runner.state_dict(),
            'optimizer': runner.optimizer.state_dict(),
            'lr_scheduler': runner.lr_scheduler.state_dict(),
        }, save_path)


class DetectionNMSHook(BaseHook):
    def __init__(self, nms_thr, pre_max_size,
                 det_key='detection', ego_only=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.nms_thr = nms_thr
        self.pre_max_size = pre_max_size
        self.nms = import_module('cosense3d.ops.iou3d_nms_utils').nms_gpu
        self.det_key = det_key
        self.ego_only = ego_only
        self.defual_pred_keys = ['box', 'scr', 'lbl', 'idx']

    def post_iter(self, runner, **kwargs):
        if self.ego_only:
            detection_out = runner.controller.data_manager.gather_ego_data(self.det_key)
        else:
            detection_out = runner.controller.data_manager.gather_cav_data(self.det_key)
        preds = []
        cav_ids = []
        for cav_id, values in detection_out.items():
            cav_ids.append(cav_id)

            out = {}
            if 'center' in values:
                out['ctr'] = values['center']
            if 'conf' in values:
                out['conf'] = values['conf']

            if 'preds' in values:
                values = values['preds']
            boxes =   values['box']
            scores =  values['scr']
            labels =  values['lbl']
            indices = values['idx']  # map index for retrieving features

            if len(values['box']) == 0:
                out.update({
                    'box': torch.zeros((0, 7), device=boxes.device),
                    'scr': torch.zeros((0,), device=scores.device),
                    'lbl': torch.zeros((0,), device=labels.device),
                    'idx': torch.zeros(indices.shape[0] if isinstance(indices, torch.Tensor) else (0,),
                                       device=indices.device),
                })
                if 'pred' in values:
                    out['pred'] = torch.zeros((0, 2, 7), device=boxes.device)
            else:
                keep = self.nms(
                    boxes[..., :7],
                    scores,
                    thresh=self.nms_thr,
                    pre_maxsize=self.pre_max_size
                )
                out.update({
                    'box': boxes[keep],
                    'scr': scores[keep],
                    'lbl': labels[keep],
                    'idx': indices[keep],
                })
                if 'pred' in values and values['pred'] is not None:
                    out['pred'] = values['pred'][keep]
                    assert len(out['pred']) != len(out['box'])
            preds.append(out)

            # from cosense3d.utils.vislib import draw_points_boxes_plt
            # points = out['ctr'].detach().cpu().numpy()
            # boxes = out['box'].detach().cpu().numpy()
            # draw_points_boxes_plt(
            #     pc_range=[-140.8, -38.4, -3.0, 140.8, 38.4, 1.0],
            #     boxes_pred=boxes,
            #     points=points,
            #     filename="/home/yuan/Pictures/tmp.png"
            # )

        runner.controller.data_manager.scatter(cav_ids, {self.det_key: preds})



class EvalDetectionBEVHook(BaseHook):
    def __init__(self, pc_range, iou_thr=[0.5, 0.7], save_result=False,
                 det_key='detection', gt_key='global_bboxes_3d', global_sort=True, **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr
        self.pc_range = pc_range
        self.save_result = save_result
        self.det_key = det_key
        self.gt_key = gt_key
        self.global_sort = global_sort
        self.result = {iou: {'tp': [], 'fp': [], 'gt': 0, 'scr': []} for iou in iou_thr}
        self.eval_funcs = import_module('cosense3d.utils.eval_detection_utils')

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'detection_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_iter(self, runner, **kwargs):
        detection = runner.controller.data_manager.gather_ego_data(self.det_key)
        gt_boxes = runner.controller.data_manager.gather_ego_data(self.gt_key)

        for i, (cav_id, preds) in enumerate(detection.items()):
            if 'preds' in preds:
                preds = preds['preds']
            preds['box'], preds['scr'], preds['lbl'], preds['idx'], preds['time'] = \
            self.filter_box_ranges(preds['box'], preds['scr'], preds['lbl'], preds['idx'], preds.get('time', None))
            cur_gt_boxes = self.filter_box_ranges(gt_boxes[cav_id])[0]
            cur_points = runner.controller.data_manager.gather_batch(i, 'points')

            if self.save_result:
                ego_key = cav_id
                senario = runner.controller.data_manager.gather_ego_data('scenario')[ego_key]
                frame = runner.controller.data_manager.gather_ego_data('frame')[ego_key]
                filename = f"{senario}.{frame}.{ego_key.split('.')[1]}.pth"
                result = {'detection': preds,
                          'gt_boxes': cur_gt_boxes,
                          'points': cur_points}
                torch.save(result, os.path.join(self.logdir, filename))

            for iou in self.iou_thr:
                self.eval_funcs.caluclate_tp_fp(
                    preds['box'][..., :7], preds['scr'], cur_gt_boxes[..., :7], self.result, iou
                )

    def filter_box_ranges(self, boxes, scores=None, labels=None, indices=None, times=None):
        mask = boxes.new_ones((len(boxes),)).bool()
        if boxes.ndim == 3:
            centers = boxes.mean(dim=1)
        else:
            centers = boxes[:, :3]
        for i in range(3):
            mask = mask & (centers[:, i] > self.pc_range[i]) & (centers[:, i] < self.pc_range[i + 3])
        boxes = boxes[mask]
        if scores is not None:
            scores = scores[mask]
        if labels is not None:
            labels = labels[mask]
        if indices is not None:
            try:
                indices = indices[mask]
            except:
                print("Number of boxes doesn't match the number of indices")
        if times is not None:
            times = times[mask]
        return boxes, scores, labels, indices, times

    def post_epoch_(self, runner, **kwargs):
        fmt_str = ("################\n"
                   "DETECTION RESULT\n"
                   "################\n")
        out_dict = self.eval_funcs.eval_final_results(
            self.result,
            self.iou_thr,
            global_sort_detections=self.global_sort
        )
        if self.global_sort:
            fmt_str += "OPV2V BEV Global Sorted:\n"
        else:
            fmt_str += "OPV2V BEV Local Sorted:\n"
        fmt_str += self.format_final_result(out_dict)
        fmt_str += "----------------\n"

        print(fmt_str)
        self.logger.log(fmt_str)

    def post_epoch(self, runner, **kwargs):
        fmt_str = ("################\n"
                   "DETECTION RESULT\n"
                   "################\n")
        fmt_str += "OPV2V BEV Global Sorted:\n"
        out_dict = self.eval_funcs.eval_final_results(
            self.result,
            self.iou_thr,
            global_sort_detections=True
        )
        fmt_str += self.format_final_result(out_dict)
        fmt_str += "OPV2V BEV Local Sorted:\n"
        out_dict = self.eval_funcs.eval_final_results(
            self.result,
            self.iou_thr,
            global_sort_detections=False
        )
        fmt_str += self.format_final_result(out_dict)
        fmt_str += "----------------\n"

        print(fmt_str)
        self.logger.log(fmt_str)


    def format_final_result(self, out_dict):
        fmt_str = ""
        for iou in self.iou_thr:
            iou_str = f"{int(iou * 100)}"
            fmt_str += f"AP@{iou_str}: {out_dict[f'ap_{iou_str}']:.3f}\n"
        return fmt_str


class EvalInfoLoggingHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'detection_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def pre_epoch(self, runner, **kwargs):
        dataset = runner.dataloader.dataset
        fmt_str = "\n" + "*" * 50 + '\n'
        fmt_str += f"Localization error : {dataset.loc_err[0]:.3f} m, {dataset.loc_err[1]:.3f} m, {np.rad2deg(dataset.loc_err[2]):.3f} degrees\n"
        fmt_str += f"Communication latency : {dataset.latency * 100:.3f} ms\n"

        spatial_alignment = runner.forward_runner.shared_modules.spatial_alignment
        fmt_str += f"Spatial alignment correction : {not spatial_alignment.disable_align}\n"

        cpm_thr = runner.controller.cav_manager.kwargs.get('share_score_thr', 0)
        fmt_str += f"CPM score threshold : {cpm_thr:.1f}\n"

        fmt_str += "*" * 50 + '\n'
        print(fmt_str)
        self.logger.log(fmt_str)


class EvalBEVSemsegHook(BaseHook):
    def __init__(self,
                 test_range,
                 test_res=0.4,
                 save_result=False,
                 eval_static=True,
                 bev_semseg_key='bev_semseg',
                 gt_bev_key='bevmap',
                 gt_boxes_key='global_bboxes_3d',
                 **kwargs):
        super().__init__(**kwargs)
        self.test_range = test_range
        self.test_res = test_res
        self.save_result = save_result
        self.eval_static = eval_static
        self.bev_semseg_key = bev_semseg_key
        self.gt_bev_key = gt_bev_key
        self.gt_boxes_key = gt_boxes_key
        self.thrs = torch.arange(0.1, 1.1, 0.1)
        self.sx = int(round((self.test_range[3] - self.test_range[0]) / self.test_res))
        self.sy = int(round((self.test_range[4] - self.test_range[1]) / self.test_res))

        self.res_dict = {
            'iou_dynamic_all': [],
            'iou_dynamic_obs': [],
            'iou_static_all': [],
            'iou_static_obs': [],
        }

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'bev_semseg_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_iter(self, runner, **kwargs):
        scene_tokens = runner.controller.data_manager.gather_ego_data('scene_tokens')
        frame = runner.controller.data_manager.gather_ego_data('frame')
        semseg = runner.controller.data_manager.gather_ego_data(self.bev_semseg_key)
        gt_bevmaps = runner.controller.data_manager.gather_ego_data(self.gt_bev_key)
        gt_boxes = runner.controller.data_manager.gather_ego_data(self.gt_boxes_key)
        for i, (cav_id, preds) in enumerate(semseg.items()):
            token = f'{scene_tokens[cav_id]}.{frame[cav_id]}'
            gt_dynamic_map = self.gt_dynamic_map(gt_boxes[cav_id])
            self.cal_ious(preds, gt_dynamic_map, 'dynamic', token)
            if self.eval_static:
                gt_static_map = self.gt_static_map(gt_bevmaps[cav_id])
                self.cal_ious(preds, gt_static_map, 'static', token)

    def cal_ious(self, preds, gt_map, tag, token=None):
        conf = self.crop_map(preds[f'conf_map_{tag}'])
        unc = self.crop_map(preds[f'unc_map_{tag}'])
        obs_mask = self.crop_map(preds[f'obs_mask_{tag}'])
        self.res_dict[f'iou_{tag}_all'].append(self.iou(conf, unc, gt_map))
        self.res_dict[f'iou_{tag}_obs'].append(self.iou(conf, unc, gt_map, obs_mask))

        if self.save_result:
            img = torch.cat([gt_map, unc, conf[..., 1]], dim=0).detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.imshow(img.T)
            plt.savefig(os.path.join(self.logdir, f'{token}.{tag}.jpg'))
            plt.close()

    def iou(self, conf, unc, gt, obs_mask=None):
        ious = []
        for thr in self.thrs:
            if obs_mask is None:
                pos_mask = torch.argmax(conf, dim=-1).bool()
                pos_mask = torch.logical_and(pos_mask, unc <= thr)
                gt_ = gt
            else:
                pos_mask = torch.argmax(conf[obs_mask], dim=-1).bool()
                pos_mask = torch.logical_and(pos_mask, unc[obs_mask] <= thr)
                gt_ = gt[obs_mask]
            mi = torch.logical_and(pos_mask, gt_).sum()
            mu = torch.logical_or(pos_mask, gt_).sum()
            ious.append(mi / mu)
        return torch.stack(ious, dim=0)

    def gt_dynamic_map(self, boxes):
        # filter box range
        mask = boxes.new_ones((len(boxes),)).bool()
        dynamic_map = torch.ones((self.sx, self.sy), device=boxes.device)
        centers = boxes[:, :3]
        for i in range(3):
            mask = mask & (centers[:, i] > self.test_range[i]) & (centers[:, i] < self.test_range[i + 3])
        boxes = boxes[mask]
        if len(boxes) > 0:
            indices = torch.stack(torch.where(dynamic_map), dim=1)
            xy = indices.float()
            xy = (xy + 0.5) * self.test_res
            xy[:, 0] += self.test_range[0]
            xy[:, 1] += self.test_range[1]
            xyz = F.pad(xy, (1, 1), 'constant', 0.0)
            boxes = F.pad(boxes, (1, 0), 'constant', 0.0)
            boxes[:, 3] = 0
            boxes_decomposed, box_idx_of_pts = points_in_boxes_gpu(
                xyz, boxes, batch_size=1
            )
            inds = indices[box_idx_of_pts >= 0].T
            dynamic_map[inds[0], inds[1]] = 0
        dynamic_map = torch.logical_not(dynamic_map)
        return dynamic_map

    def gt_static_map(self, bevmap):
        # map has higher resolution, downsample 2x
        # bevmap = torch.flip(bevmap, [0])
        return bevmap[::2, ::2]

    def crop_map(self, bevmap):
        sx, sy = bevmap.shape[:2]
        sx_crop = (sx - self.sx) // 2
        sy_crop = (sy - self.sy) // 2
        return bevmap[sx_crop:-sx_crop, sy_crop:-sy_crop]

    def post_epoch(self, runner, **kwargs):
        fmt_str = ("#################\n"
                   "BEV SEMSEG RESULT\n"
                   "#################\n")
        fmt_str += f"{'thr':18s} |  " + "  ".join([f"{v:4.1f} " for v in self.thrs]) + "\n"
        fmt_str += "-" * (23 + 70) + "\n"
        for k, vs in self.res_dict.items():
            vs = torch.stack(vs, dim=0).mean(dim=0) * 100
            if isinstance(vs, int):
                continue
            s1 = f"{k:18s} |  "
            if isinstance(vs, float):
                s2 = f"{vs:4.1f} \n"
            else:
                s2 = "  ".join([f"{v:4.1f} " for v in vs]) + "\n"
            fmt_str += s1 + s2
        print(fmt_str)
        self.logger.log(fmt_str)


class SaveResultHook(BaseHook):
    def __init__(self, save_keys=['detection_local'], **kwargs):
        super().__init__(**kwargs)
        self.keys = save_keys

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'addtional_result')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_iter(self, runner, **kwargs):
        dm = runner.controller.data_manager
        data_dict = {}
        for k in self.keys:
            data_dict[k] = dm.gather_cav_data(k)

        cav_id = runner.controller.cav_manager.cavs[0][0].id
        scenario = dm.gather_ego_data('scenario')[cav_id]
        frame = dm.gather_ego_data('frame')[cav_id]
        filename = f"{scenario}.{frame}.{cav_id.split('.')[1]}.pth"
        torch.save(data_dict, os.path.join(self.logdir, filename))










