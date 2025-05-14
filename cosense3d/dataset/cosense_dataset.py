import copy
import os
import random
from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from cosense3d.dataset.pipeline import Pipeline
from cosense3d.utils.misc import load_json


class CosenseDataset(Dataset):
    LABEL_COLORS = {}
    VALID_CLS = []

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode
        self.COM_RANGE = self.cfgs.get('com_range', 70)
        self.latency = cfgs.get('latency', 0)
        self.loc_err = np.array(cfgs.get('loc_err', [0, 0, 0]))
        if cfgs.get('enable_split_sub_folder', True):
            self.data_path = os.path.join(self.cfgs['data_path'], self.mode)
        else:
            self.data_path = self.cfgs['data_path']

        self.max_num_cavs = cfgs['max_num_cavs']

        self.init_dataset()

        self.pipeline = Pipeline(cfgs[f'{mode}_pipeline'])
        # for frames that do not need loss calculation, omit gt-loading to save time
        if 'inf_pipeline' in cfgs:
            self.inf_pipeline = Pipeline(cfgs['inf_pipeline'])
        else:
            self.inf_pipeline = self.pipeline

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.load_frame_data(item)

    def load_frame_data(self,
                        item: int,
                        prev_agents: Optional[List] = None,
                        prev_item: Optional[int] = None,
                        omit_gt: Optional[bool] = False,
                        loc_err: Union[list, None] = None) -> dict:
        """
        Load all data and annotations from one frame to standard CoSense format.

        :param item: sample index.
        :param prev_agents: only load data the previous agents if given, this is used for temporal data loading.
        :param prev_item: the index of the previous loaded sample.
        :param omit_gt: whether to omit loading the ground truth annotations.
        :param loc_err: localization error.
        :return: data_dict
        """
        sample_info = self.load_sample_info(item, prev_agents, prev_item)
        sample_info['loc_err'] = [loc_err[i] for i in range(len(sample_info['valid_agent_ids']))]
        if omit_gt:
            data_dict = self.inf_pipeline(sample_info)
        else:
            data_dict = self.pipeline(sample_info)
        data_dict.pop('sample_info')
        data_dict.pop('data_path')
        return data_dict

    def init_dataset(self):
        """Load all necessary meta information"""
        self.load_meta()
        self.parse_samples()

    def parse_samples(self):
        """List all frame-wise instances"""
        # list all frames, each frame as a sample
        self.samples = []
        drop_scenarios = self.cfgs.get('drop_scenarios', [])
        for scenario, scontent in self.meta_dict.items():
            if scenario in drop_scenarios:
                continue
            self.samples.extend(sorted([[scenario, frame] for frame in scontent.keys()]))
        self.samples = sorted(self.samples)

        print(f"{self.mode} : {len(self.samples)} samples.")

    def load_meta(self):
        """Load meta data from CoSense json files"""
        self.meta_dict = {}
        meta_dir = self.cfgs['meta_path']
        if meta_dir == '':
            return
        if 'split' in self.cfgs:
            scenarios = self.cfgs['split'][self.mode]
        elif os.path.exists(os.path.join(self.cfgs['meta_path'], f"{self.mode}.txt")):
            with open(os.path.join(self.cfgs['meta_path'], f"{self.mode}.txt"), 'r') as fh:
                scenarios = [l.strip() for l in fh.readlines() if len(l.strip()) > 0]
        else:
            scenarios = [d[:-5] for d in os.listdir(meta_dir) if 'json' in d]

        for scenario in scenarios:
            meta_file = os.path.join(meta_dir, f"{scenario}.json")
            scenario_dict = load_json(meta_file)
            # if '017343' not in scenario_dict:
            #     continue
            # scenario_dict = {s: scenario_dict[s] for s in list(scenario_dict.keys())[:1]}
            self.meta_dict[scenario] = scenario_dict

    def load_sample_info(self, item: int, prev_agents: Optional[List] = None, prev_item: Optional[int] = None) -> dict:
        """
        Load meta info of the ```item```'th sample.

        :param item: sample index.
        :param prev_agents: only load data the previous agents if given, this is used for temporal data loading.
        :param prev_item: the index of the previous loaded sample.
        :return: batch_dict: dict(scenario: str, frame: str, sample_info: dict)
        """
        # load meta info
        scenario, frame = self.samples[item]
        sample_info = copy.deepcopy(self.meta_dict[scenario][frame])

        if prev_item is None:
            prev_item = max(item - 1, 0)
        prev_scenario, prev_frame = self.samples[prev_item]
        prev_idx = f'{prev_scenario}.{prev_frame}'
        next_item = min(item + 1, self.__len__() - 1)
        next_scenario, next_frame = self.samples[next_item]
        next_idx = f'{next_scenario}.{next_frame}'

        if prev_scenario != scenario:
            prev_agents = None
        valid_agent_ids = self.get_valid_agents(sample_info, prev_agents)

        # previous agents might not in current frame when load sequential data
        scenario_tokens = [f'{scenario}.{ai}' for ai in valid_agent_ids if ai in sample_info['agents']]

        # if latency > 0, set the sample info of coop. cavs to previous frame at -latency
        if self.latency != 0:
            # get random latency if latency flag is negative
            latency = np.random.randint(abs(self.latency) + 1) if self.latency < 0 else self.latency
            latent_item = max(item - latency, 0)
            try:
                latent_scenario, latent_frame = self.samples[latent_item]
            except:
                print(latent_item)
            if latent_scenario != scenario:
                # make sure the scenario is the same as the current frame
                latent_scenario = scenario
                latent_frame = frame
            latent_info = copy.deepcopy(self.meta_dict[latent_scenario][latent_frame])
            # update coop agent info to latent frame
            for cav_id in valid_agent_ids:
                if cav_id == sample_info['meta']['ego_id']:
                    continue
                if cav_id in latent_info['agents']:
                    sample_info['agents'][cav_id] = latent_info['agents'][cav_id]

        return {
            'scenario': scenario,
            'frame': frame,
            'data_path': self.data_path,
            'sample_info': sample_info,
            'valid_agent_ids': valid_agent_ids,
            'scene_tokens': scenario_tokens,
        }

    def get_valid_agents(self, sample_info: dict, prev_agents: Optional[List] = None) -> List:
        """
        Return prev_agents if given else select the given number of agents in the communication range
         which includes the ego agent.

        Parameters
        ----------
        sample_info: meta info the one sample.
        prev_agents: list of the agent ids loader last time.

        Returns
        -------
        agents_ids: list of valid agent for the current sample
        """
        if prev_agents is not None:
            return prev_agents
        else:
            agents = sample_info['agents']
            ego_id = str(sample_info['meta']['ego_id'])
            agents_ids = [ego_id]
            # filter cavs in communication range
            ego_pose_vec = agents[ego_id]['pose']
            in_range_cavs = []
            for ai, adict in agents.items():
                if ai == ego_id:
                    continue
                if ((adict['pose'][0] - ego_pose_vec[0])**2 + (adict['pose'][1] - ego_pose_vec[1])**2
                        < self.COM_RANGE**2):
                    in_range_cavs.append(ai)
            if self.max_num_cavs > 1:
                agents_ids += random.sample(in_range_cavs, k=min(self.max_num_cavs - 1, len(in_range_cavs)))
        return agents_ids

    @staticmethod
    def collate_batch(batch_list):
        keys = batch_list[0].keys()
        batch_dict = {k:[] for k in keys}

        def list_np_to_tensor(ls):
            ls_tensor = []
            for i, l in enumerate(ls):
                if isinstance(l, list):
                    l_tensor = list_np_to_tensor(l)
                    ls_tensor.append(l_tensor)
                elif isinstance(l, np.ndarray):
                    tensor = torch.from_numpy(l)
                    if l.dtype == np.float64:
                        tensor = tensor.float()
                    ls_tensor.append(tensor)
                else:
                    ls_tensor.append(l)
            return ls_tensor

        for k in keys:
            if isinstance(batch_list[0][k], np.ndarray):
                batch_dict[k] = [torch.from_numpy(batch[k]) for batch in batch_list]
            elif isinstance(batch_list[0][k], list):
                batch_dict[k] = [list_np_to_tensor(batch[k]) for batch in batch_list]
            else:
                batch_dict[k] = [batch[k] for batch in batch_list]
        return batch_dict


if __name__=="__main__":
    from cosense3d.utils.misc import load_yaml
    from torch.utils.data import DataLoader
    cfgs = load_yaml("/mars/projects20/CoSense3D/cosense3d/config/petr.yaml")
    cosense_dataset = CosenseDataset(cfgs['DATASET'], 'train')
    cosense_dataloader = DataLoader(dataset=cosense_dataset, collate_fn=cosense_dataset.collate_batch)
    for data in cosense_dataloader:
        print(data.keys())