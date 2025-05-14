import random
import numpy as np
from cosense3d.dataset.cosense_dataset import CosenseDataset


class TemporalCosenseDataset(CosenseDataset):
    """Sequential Cosense data loader."""
    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)
        self.seq_len = cfgs['seq_len']
        self.n_loss_frame = cfgs.get('n_loss_frame', 1)
        self.rand_len = cfgs.get('rand_len', 0)
        self.seq_mode = cfgs.get('seq_mode', False)
        self.clean_seq = cfgs.get('clean_seq', False)

    def __getitem__(self, index):
        queue = []
        index_list = list(range(index - self.seq_len - self.rand_len + 1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.rand_len:])
        index_list.append(index)
        prev_scene_token = None
        prev_agents = None
        prev_i = None
        num_cav = None
        omit_gt = [True] * (len(index_list) - self.n_loss_frame) + [False] * self.n_loss_frame
        loc_err = np.random.randn(self.max_num_cavs, 3) * self.loc_err.reshape(-1, 3)

        for i, idx in enumerate(index_list):
            idx = max(0, idx)
            input_dict = self.load_frame_data(
                idx, prev_agents, prev_i, omit_gt=omit_gt[i], loc_err=loc_err)
            prev_i = idx

            if not self.seq_mode:  # for sliding window only
                prev_exists = []
                prev_agents = []
                for tk in input_dict['scene_tokens']:
                    prev_agents.append(tk.split('.')[-1])
                    if prev_scene_token is not None and tk in prev_scene_token:
                        prev_exists.append(np.array([True]))
                    else:
                        prev_exists.append(np.array([False]))
                input_dict.update(dict(prev_exists=prev_exists))
                prev_scene_token = input_dict['scene_tokens']

            queue.append(input_dict)

        # remove frames not belong to the current sequence
        # and ensure all frames have the same ego id
        valid_idx_start = 0
        if self.clean_seq:
            ego_id = queue[-1]['valid_agent_ids'][0]
            for idx in range(len(queue)):
                if queue[idx]['valid_agent_ids'][0] != ego_id:
                    valid_idx_start = idx + 1
        queue = {k: [q[k] if k in q else None for q in queue[valid_idx_start:]] for k in queue[-1].keys()}
        return queue


if __name__=="__main__":
    from cosense3d.utils.misc import load_yaml
    from torch.utils.data import DataLoader
    cfgs = load_yaml("/mars/projects20/CoSense3D/cosense3d/config/petr.yaml")
    cosense_dataset = TemporalCosenseDataset(cfgs['DATASET'], 'train')
    cosense_dataloader = DataLoader(dataset=cosense_dataset, collate_fn=cosense_dataset.collate_batch)
    for data in cosense_dataloader:
        print(data.keys())