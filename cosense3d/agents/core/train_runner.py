
import os, glob, logging
from datetime import datetime

from torch.nn.parallel import DistributedDataParallel as DDP

from cosense3d.utils.train_utils import *
from cosense3d.utils.lr_scheduler import build_lr_scheduler
from cosense3d.utils.logger import LogMeter
from cosense3d.utils.misc import ensure_dir
from cosense3d.agents.core.base_runner import BaseRunner
from cosense3d.agents.utils.deco import save_ckpt_on_error


class TrainRunner(BaseRunner):
    def __init__(self,
                 max_epoch,
                 optimizer,
                 lr_scheduler,
                 gpus=0,
                 resume_from=None,
                 load_from=None,
                 run_name='default',
                 log_dir='work_dir',
                 use_wandb=False,
                 debug=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.gpus = gpus
        self.gpu_id = 0
        self.dist = False
        self.debug = debug
        if gpus > 0:
            self.dist = True
            self.gpu_id = int(os.environ.get("LOCAL_RANK", 0))
            self.forward_runner.to_gpu(self.gpu_id)
            self.forward_runner = DDP(self.forward_runner, device_ids=[self.gpu_id])
        self.optimizer = build_optimizer(self.forward_runner, optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, lr_scheduler,
                                               len(self.dataloader))
        self.total_epochs = max_epoch
        self.start_epoch = 1

        self.resume(resume_from, load_from)
        self.setup_logger(resume_from, run_name, log_dir, use_wandb)

    def setup_logger(self, resume_from, run_name, log_dir, use_wandb):
        if resume_from is not None:
            if os.path.isfile(resume_from):
                log_path = os.path.dirname(resume_from)
            else:
                log_path = resume_from
        else:
            now = datetime.now().strftime('%m-%d-%H-%M-%S')
            run_name = run_name + '_' + now
            log_path = os.path.join(log_dir, run_name)
            ensure_dir(log_path)
        wandb_project_name = run_name if use_wandb else None
        self.logger = LogMeter(self.total_iter, log_path, log_every=self.log_every,
                               wandb_project=wandb_project_name)

    def resume(self, resume_from, load_from):
        if resume_from is not None or load_from is not None:
            load_path = resume_from if resume_from is not None else load_from
            assert os.path.exists(load_path), f'resume/load path does not exist: {resume_from}.'
            if os.path.isdir(load_path):
                ckpts = glob.glob(os.path.join(load_path, 'epoch*.pth'))
                if len(ckpts) > 0:
                    epochs = [int(os.path.basename(ckpt)[5:-4]) for ckpt in ckpts]
                    max_idx = epochs.index(max(epochs))
                    ckpt = ckpts[max_idx]
                elif os.path.exists(os.path.join(load_path, 'last.pth')):
                    ckpt = os.path.join(load_path, 'last.pth')
                else:
                    raise IOError(f'No checkpoint found in directory {load_path}.')
            elif os.path.isfile(load_path):
                ckpt = load_path
            else:
                raise IOError(f'Failed to load checkpoint from {load_path}.')
            logging.info(f"Resuming the model from checkpoint: {ckpt}")
            ckpt = torch.load(ckpt)
            load_model_dict(self.forward_runner, ckpt['model'], self.dist)
            if resume_from is not None:
                self.start_epoch = ckpt['epoch'] + 1
                self.epoch = ckpt['epoch'] + 1
                if 'lr_scheduler' in ckpt:
                    self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                try:
                    if 'optimizer' in ckpt:
                        self.optimizer.load_state_dict(ckpt['optimizer'])
                except:
                    warnings.warn("Cannot load optimizer state_dict, "
                                  "there might be training parameter changes, "
                                  "please consider using 'load-from'.")

    def run(self):
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.start_epoch, self.total_epochs + 1):
                self.hooks(self, 'pre_epoch')
                self.run_epoch()
                self.hooks(self, 'post_epoch')
                self.lr_scheduler.step_epoch(i)
                self.epoch += 1
                self.iter = 1

    def step(self):
        data = self.next_batch()
        self.run_itr(data)

    def run_epoch(self):
        if self.dist:
            self.dataloader.sampler.set_epoch(self.epoch)
        for data in self.dataloader:
            # print(f'{self.gpu_id}: run_itr{self.iter}: 0')
            self.hooks(self, 'pre_iter')
            self.run_itr(data)
            self.hooks(self, 'post_iter')

    @save_ckpt_on_error
    def run_itr(self, data):
        load_tensors_to_gpu(data, self.gpu_id)
        self.optimizer.zero_grad()
        total_loss, loss_dict = self.controller.train_forward(
            data, epoch=self.epoch, itr=self.iter, gpu_id=self.gpu_id)
        total_loss.backward()

        grad_norm = clip_grads(self.controller.parameters)
        loss_dict['grad_norm'] = grad_norm
        # Updating parameters
        self.optimizer.step()

        self.lr_scheduler.step_itr(self.iter + self.epoch * self.total_iter)

        if self.logger is not None and self.gpu_id == 0:
            # rec_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
            rec_lr = self.lr_scheduler.get_last_lr()[0]
            self.logger.log(self.epoch, self.iter, rec_lr, **loss_dict)

        del data
        self.iter += 1



