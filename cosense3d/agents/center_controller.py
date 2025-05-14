import matplotlib.pyplot as plt
import torch

from cosense3d.agents import core


class CenterController:
    def __init__(self, cfg, data_loader, dist=False):
        self.mode = data_loader.dataset.mode
        self.dist = dist
        self.seq_len = data_loader.dataset.seq_len
        self.data_info = data_loader.dataset.cfgs['data_info']
        self.num_loss_frame = cfg.get('num_loss_frame', 1)
        self.batch_seq = cfg.get('batch_seq', False)
        self.setup_core(cfg)
        self.global_data = {}

    def setup_core(self, cfg):
        if self.batch_seq:
            cav_manager = core.SeqCAVManager
            data_manager = core.SeqDataManager
            task_manager = core.SeqTaskManager(self.seq_len)

        else:
            cav_manager = core.CAVManager
            data_manager = core.DataManager
            task_manager = core.TaskManager()
        cfg['cav_manager']['seq_len'] = self.seq_len
        self.cav_manager = cav_manager(**self.update_cfg(cfg['cav_manager'],
                                                             self.data_info))
        self.data_manager = data_manager(
            self.cav_manager, **self.update_cfg(
                cfg['data_manager'][self.mode], self.data_info))
        self.task_manager = task_manager
        self.forward_runner = core.ForwardRunner(cfg['shared_modules'],
                                                 self.data_manager,
                                                 self.dist, **cfg.get('forward_runner', {}))

    def update_cfg(self, cfg, *args):
        for arg in args:
            cfg.update(arg)
        return cfg

    @property
    def modules(self):
        return self.forward_runner.shared_modules

    @property
    def model(self):
        return self.forward_runner

    @property
    def parameters(self):
        return self.forward_runner.parameters()

    def train_forward(self, batch_dict, **kwargs):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        self.data_manager.add_loc_err(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        self.cav_manager.reset()

        if self.batch_seq:
            return self.run_seq(seq_data, training_mode=True, **kwargs)
        else:
            loss = 0
            loss_dict = {}
            for i, data in enumerate(seq_data): # a few seqs from dataloader might < self.seq_lens
                with_loss = i >= self.seq_len - self.num_loss_frame
                kwargs['seq_idx'] = i
                frame_loss_dict = self.run_frame(data, with_loss, training_mode=True, **kwargs)
                for k, v in frame_loss_dict.items():
                    if 'loss' in k:
                        loss = loss + v
                    loss_dict[f'f{i}.{k}'] = v
            loss_dict['total_loss'] = loss
            return loss, loss_dict

    def test_forward(self, batch_dict, **kwargs):
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        self.data_manager.add_loc_err(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        self.cav_manager.reset()

        for i in range(self.seq_len):
            kwargs['seq_idx'] = i
            self.run_frame(seq_data[i],
                           with_loss=False,
                           training_mode=False,
                           **kwargs)

    def vis_forward(self, batch_dict, **kwargs):
        kwargs['seq_idx'] = 0
        self.data_manager.generate_augment_params(batch_dict, self.seq_len)
        self.data_manager.add_loc_err(batch_dict, self.seq_len)
        seq_data = self.data_manager.distribute_to_seq_list(batch_dict, self.seq_len)
        frame_data = seq_data[0]
        self.cav_manager.update_cav_info(**frame_data)
        self.data_manager.distribute_to_cav(**frame_data)
        # send and receive request
        request = self.cav_manager.send_request()
        self.cav_manager.receive_request(request)
        # apply data online transform
        self.cav_manager.forward(False, False, **kwargs)

    def run_frame(self, frame_data, with_loss, training_mode, **kwargs):
        self.cav_manager.update_cav_info(**frame_data)
        self.data_manager.distribute_to_cav(**frame_data)
        self.cav_manager.apply_cav_function('pre_update_memory')

        # get pseudo forward tasks
        tasks = self.cav_manager.forward(with_loss, training_mode, **kwargs)
        batched_tasks = self.task_manager.summarize_tasks(tasks)

        # send and receive request
        request = self.cav_manager.send_request()
        self.cav_manager.receive_request(request)

        # prepare local data
        self.cav_manager.apply_cav_function('prepare_data')

        # correct localization errors
        self.forward_runner(batched_tasks[0]['no_grad'], with_grad=False, **kwargs)
        self.forward_runner(batched_tasks[0]['with_grad'], with_grad=True, **kwargs)

        # apply data transformation with the corrected localization
        self.cav_manager.apply_cav_function('transform_data')

        # preprocess after transformation
        self.data_manager.apply_preprocess()
        # self.data_manager.vis_global_data_plt(['vis_ref_pts', 'vis_poses'], kwargs['seq_idx'] + 1)

        # from cosense3d.utils.vislib import plot_cavs_points
        # plot_cavs_points(self.cav_manager.cavs[0])

        # process local cav data
        self.forward_runner(batched_tasks[1]['no_grad'], with_grad=False, **kwargs)
        self.forward_runner(batched_tasks[1]['with_grad'], with_grad=training_mode, **kwargs)

        # send coop cav feature-level cpm to ego cav
        response = self.cav_manager.send_response()
        self.cav_manager.receive_response(response)

        # process ego cav data and fuse data from coop cav with grad if training
        self.forward_runner(batched_tasks[2]['with_grad'], with_grad=training_mode, **kwargs)
        self.forward_runner(batched_tasks[2]['no_grad'], with_grad=False, **kwargs)
        self.cav_manager.apply_cav_function('post_update_memory')

        frame_loss_dict = {}
        if with_loss:
            frame_loss_dict = self.forward_runner.frame_loss(batched_tasks[3]['loss'], **kwargs)
        return frame_loss_dict

    def run_seq(self, seq_data, training_mode, **kwargs):
        cur_len = len(seq_data)
        self.cav_manager.update_cav_info(seq_data)
        self.data_manager.distribute_to_cav(seq_data)
        self.cav_manager.apply_cav_function('init_memory')

        # send and receive request
        request = self.cav_manager.send_request()
        self.cav_manager.receive_request(request)
        # get pseudo forward tasks
        tasks = self.cav_manager.forward(training_mode, self.num_loss_frame, cur_len)
        batched_tasks = self.task_manager.summarize_tasks(tasks)
        # preprocess after transformation to ego frame
        self.data_manager.apply_preprocess()

        # process local cav data
        if 'no_grad' in batched_tasks[0] and len(batched_tasks[0]['no_grad']) > 0:
            self.forward_runner(batched_tasks[0]['no_grad'], with_grad=False, **kwargs)

        self.forward_runner(batched_tasks[0]['with_grad'], with_grad=training_mode, **kwargs)

        # process tasks that needs to be run sequentially
        seq_tasks = self.task_manager.parallel_to_sequential(batched_tasks[1])
        for i in range(cur_len):
            self.cav_manager.apply_cav_function('pre_update_memory', seq_idx=i)
            if 'no_grad' in seq_tasks and len(seq_tasks['no_grad'][i]) > 0:
                self.forward_runner(seq_tasks['no_grad'][i], with_grad=False, **kwargs)
            self.forward_runner(seq_tasks['with_grad'][i], with_grad=training_mode, **kwargs)
            self.cav_manager.apply_cav_function('post_update_memory', seq_idx=i)

        # send coop cav feature-level cpm to ego cav
        response = self.cav_manager.send_response()
        self.cav_manager.receive_response(response)

        if 2 not in batched_tasks:
            print([d['valid_agent_ids'] for d in seq_data])
        # process ego cav data and fuse data from coop cav with grad if training
        self.forward_runner(batched_tasks[2]['with_grad'], with_grad=training_mode, **kwargs)
        if 'no_grad' in batched_tasks[2]:
            self.forward_runner(batched_tasks[2]['no_grad'], with_grad=False, **kwargs)
        loss, loss_dict = self.forward_runner.loss(batched_tasks[3]['loss'], with_grad=False, **kwargs)
        return loss, loss_dict







