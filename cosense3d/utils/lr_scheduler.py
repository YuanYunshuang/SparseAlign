from torch.optim import lr_scheduler as torch_lr
from torch.optim import Optimizer


def build_lr_scheduler(optimizer, cfg, total_iter):
    return LRUpdater(optimizer, total_iter, **cfg)


class TransformerAdaptiveScheduler(torch_lr._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 itrs_per_epoch: int,
                 last_epoch: int = -1,
                 global_fade_ratio: float = 1,
                 verbose: bool=False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        self.global_fade_ratio = global_fade_ratio
        super().__init__(optimizer, last_epoch)
        if last_epoch > 0:
            self._step_count = itrs_per_epoch * last_epoch

    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.dim_embed, self.warmup_steps) * self.global_fade_ratio
        return [lr] * self.num_param_groups

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


class LRUpdater:
    """
    Unified API for updating LR with different LR schedulers.
    """
    def __init__(self, optimizer, total_iter, policy, **kwargs):
        self.policy = policy
        self.total_itr = total_iter
        if policy == 'MultiStepLR':
            # construct a learning rate scheduler
            self.lr_scheduler = torch_lr.MultiStepLR(optimizer, **kwargs)
        elif policy == 'CosineAnnealingWarm':
            from timm.scheduler.cosine_lr import CosineLRScheduler
            num_steps = kwargs['epochs'] * total_iter
            warmup_lr = kwargs['warmup_lr']
            warmup_steps = kwargs['warmup_epochs'] * total_iter
            lr_min = kwargs['lr_min']
            decay_rate = kwargs.get('decay_rate', 0.5)

            self.lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=lr_min,
                warmup_lr_init=warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                cycle_decay=decay_rate
            )
        elif policy == 'TransformerAdaptiveScheduler':
            kwargs['itrs_per_epoch'] = total_iter
            self.lr_scheduler = TransformerAdaptiveScheduler(optimizer, **kwargs)
        else:
            raise NotImplementedError

        self.optimizer = self.lr_scheduler.optimizer

    def step_epoch(self, epoch):
        if self.policy == 'TransformerAdaptiveScheduler':
            pass
        elif self.policy in ['CosineAnnealingWarm',]:
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()

    def step_itr(self, itr):
        if self.policy == 'TransformerAdaptiveScheduler':
            self.lr_scheduler.step()

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()


if __name__=="__main__":
    import torch
    import matplotlib.pyplot as plt
    params = torch.nn.Parameter(torch.rand(10, 10))
    optimizer = torch.optim.AdamW([params],
                                  lr=0.0002,
                                  weight_decay=1e-2,
                                  betas=(0.9, 0.98),
                                  eps=1.0e-9,
                                  # init_lr=0.001,
                                  )
    lr_scheduler = TransformerAdaptiveScheduler(
        optimizer,
        dim_embed=256,
        warmup_steps=3000,
        itrs_per_epoch=2000,
        last_epoch=-1,
        global_fade_ratio=0.5
    )

    # torch.save(optimizer.state_dict(), 'optimizer_checkpoint.pth')
    # optimizer.load_state_dict(torch.load('optimizer_checkpoint.pth'))
    # lr_scheduler = TransformerAdaptiveScheduler(
    #     optimizer,
    #     dim_embed=256,
    #     warmup_steps=4000,
    #     itrs_per_epoch=2000,
    #     last_epoch=3,
    # )

    lrs = []
    for epoch in range(50 * 2000):
        lrs.append(lr_scheduler.get_lr()[0])
        optimizer.step()
        lr_scheduler.step()

    plt.plot(torch.arange(len(lrs)).numpy(), lrs)
    plt.show()
    plt.close()
