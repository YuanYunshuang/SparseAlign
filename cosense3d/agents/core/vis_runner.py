
import os, glob, logging
from tqdm import tqdm

from cosense3d.utils.train_utils import *
from cosense3d.agents.core.base_runner import BaseRunner


class VisRunner(BaseRunner):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.progress_bar = tqdm(total=self.total_iter)

    def load(self, load_from):
        assert load_from is not None, "load path not given."
        assert os.path.exists(load_from), f'resume path does not exist: {load_from}.'
        if os.path.isfile(load_from):
            ckpt = load_from
        else:
            ckpts = glob.glob(os.path.join(load_from, 'epoch*.pth'))
            if len(ckpts) > 0:
                epochs = [int(os.path.basename(ckpt)[5:-4]) for ckpt in ckpts]
                max_idx = epochs.index(max(epochs))
                ckpt = ckpts[max_idx]
            elif os.path.exists(os.path.join(load_from, 'last.pth')):
                ckpt = os.path.join(load_from, 'last.pth')
            else:
                raise IOError('No checkpoint found.')
        logging.info(f"Resuming the model from checkpoint: {ckpt}")
        ckpt_dict = torch.load(ckpt)
        load_model_dict(self.forward_runner, ckpt_dict['model'])
        return ckpt

    def run(self):
        for data in self.dataloader:
            self.run_itr(data)
        self.progress_bar.close()

    def step(self):
        data = self.next_batch()
        self.run_itr(data)

    def run_itr(self, data):
        self.hooks(self, 'pre_iter')
        if data['scenario'][0][0] == '10.0' and data['frame'][0][0] == '018076':
            print('d')
        load_tensors_to_gpu(data)
        self.controller.vis_forward(data)

        self.hooks(self, 'post_iter')
        self.iter += 1
        self.progress_bar.update(1)





