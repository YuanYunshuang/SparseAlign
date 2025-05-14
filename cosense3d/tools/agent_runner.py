import os, sys
import argparse
import logging

import numpy as np
import torch

os.system('ulimit -n 2048')
os.environ['OMP_NUM_THREADS'] = "16" # for ME

from cosense3d.dataset import get_dataloader
from cosense3d.utils.misc import setup_logger
from cosense3d.config import load_config, save_config
from cosense3d.utils.train_utils import seed_everything
from cosense3d.agents.center_controller import CenterController
from cosense3d.agents.core.train_runner import TrainRunner
from cosense3d.agents.core.test_runner import TestRunner
from cosense3d.agents.core.vis_runner import VisRunner
from cosense3d.tools.path_cfgs import parse_paths


def ddp_setup():
    from torch.distributed import init_process_group
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class AgentRunner:
    def __init__(self, args, cfgs):
        self.visualize = args.visualize or 'vis' in args.mode
        self.mode = args.mode
        if args.gpus > 0:
            self.dist = True
            ddp_setup()
        else:
            self.dist = False
        if self.visualize:
            from cosense3d.agents.core.gui import GUI
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication(sys.argv)
            self.gui = GUI(args.mode, cfgs['VISUALIZATION'])

        self.build_runner(args, cfgs)

    def build_runner(self, args, cfgs):
        dataloader = get_dataloader(cfgs['DATASET'],
                                    args.mode.replace('vis_', ''),
                                    self.dist)
        center_controller = CenterController(cfgs['CONTROLLER'], dataloader, self.dist)
        if args.mode == 'train':
            self.runner = TrainRunner(dataloader=dataloader,
                                      controller=center_controller,
                                      **cfgs['TRAIN'])
        elif args.mode == 'test':
            self.runner = TestRunner(dataloader=dataloader,
                                     controller=center_controller,
                                     **cfgs['TEST'])
        else:
            self.runner = VisRunner(dataloader=dataloader,
                                    controller=center_controller,)

    def visible_run(self):
        self.gui.setRunner(self.runner)
        self.app.installEventFilter(self.gui)

        # self.app.setStyle("Fusion")
        from PyQt5.QtWidgets import QDesktopWidget
        desktop = QDesktopWidget().availableGeometry()
        width = (desktop.width() - self.gui.width()) / 2
        height = (desktop.height() - self.gui.height()) / 2

        self.gui.move(int(width), int(height))
        self.gui.initGUI()
        # Start GUI
        self.gui.show()

        logging.info("Showing GUI...")
        sys.exit(self.app.exec_())

    def run(self):
        try:
            if self.visualize:
                self.visible_run()
            else:
                self.runner.run()
        finally:
            self.runner.logger.close()
            if self.dist:
                from torch.distributed import destroy_process_group
                destroy_process_group()


def parse_cfgs(args):
    if args.run_name is None:
        args.run_name = os.path.basename(args.config).split('.')[0]
    cfgs = load_config(args)
    if args.gpus:
        cfgs['TRAIN']['gpus'] = args.gpus
    if args.batch_size is not None:
        cfgs['DATASET']['batch_size_train'] = args.batch_size
    if args.n_workers is not None:
        cfgs['DATASET']['n_workers'] = args.n_workers
    if args.meta_path is not None:
        cfgs['DATASET']['meta_path'] = args.meta_path
    if args.data_path is not None:
        cfgs['DATASET']['data_path'] = args.data_path
    if args.data_latency is not None:
        cfgs['DATASET']['latency'] = args.data_latency
    if args.loc_err is not None:
        loc_err = [float(x) for x in args.loc_err.split(',')]
        cfgs['DATASET']['loc_err'] = [loc_err[0], loc_err[1], np.deg2rad(loc_err[2])]
    if args.cnt_cpm_size:
        cfgs['TEST']['hooks'].append({'type': 'CPMStatisticHook'})
        cfgs['CONTROLLER']['cav_manager']['cpm_statistic'] = True
    if args.cpm_thr is not None:
        cfgs['CONTROLLER']['cav_manager']['share_score_thr'] = args.cpm_thr
    elif 'share_score_thr' not in cfgs['CONTROLLER']['cav_manager']:
        cfgs['CONTROLLER']['cav_manager']['share_score_thr'] = 0.0
    if args.fsa:
        cfgs['CONTROLLER']['cav_manager']['FSA'] = True
    if args.save_result:
        for hook in cfgs['TEST']['hooks']:
            if hook['type'] == 'EvalDetectionBEVHook':
                hook['save_result'] = args.save_result
    if args.save_local_det:
        cfgs['TEST']['hooks'].append({
            'type': 'SaveResultHook',
            'save_keys': ['detection_local']
        })
    if args.align:
        cfgs['CONTROLLER']['shared_modules']['spatial_alignment']['disable_align'] = False
    cfgs['DATASET']['max_num_cavs'] = args.max_num_cavs
    cfgs['DATASET']['seq_len'] = args.seq_len
    parse_paths(cfgs)
    return cfgs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--mode", type=str, default="test",
                        help="train | test | vis_train | vis_test | train_test")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--resume-from", type=str)
    parser.add_argument("--load-from", type=str)
    parser.add_argument("--log-dir", type=str, default=f"{os.path.dirname(__file__)}/../../logs")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--meta-path", type=str)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-workers", type=int)
    parser.add_argument("--data-latency", type=int,
                        help="-1: random latency selected from (0, 1, 2)*100ms;\n"
                             " 0: coop. data has no additional latency relative to ego frame;\n"
                             " n>0: coop. data has n*100ms latency relative to ego frame.")
    parser.add_argument("--loc-err", type=str,
                        help="localization errors for x, y translation "
                             "and rotation angle along z-axis."
                             "example: `0.5,0.5,1` for 0.5m deviation at x and y axis "
                             "and 1 degree rotation angle")
    parser.add_argument("--cnt-cpm-size", action="store_true")
    parser.add_argument("--cpm-thr", type=float)
    parser.add_argument("--save-result", action="store_true")
    parser.add_argument("--save-local-det", action="store_true")
    parser.add_argument("--max-num-cavs", type=int, default=7)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--fsa", action="store_true")
    parser.add_argument("--align", action="store_true")
    args = parser.parse_args()

    setup_logger(args.run_name, args.debug)
    # if 'vis' in args.mode:
    #     args.config = "./config/defaults/base_cav.yaml"

    seed_everything(args.seed)
    if args.mode == "train":
        cfgs = parse_cfgs(args)
        agent_runner = AgentRunner(args, cfgs)
        save_config(cfgs, agent_runner.runner.logdir)
        agent_runner.run()
    elif args.mode == "train_test":
        args.mode = "train"
        cfgs = parse_cfgs(args)
        agent_runner = AgentRunner(args, cfgs)
        save_config(cfgs, agent_runner.runner.logdir)
        agent_runner.run()
        args.mode = "test"
        args.load_from = agent_runner.runner.logdir
        args.data_latency = 0
        cfgs = parse_cfgs(args)
        agent_runner = AgentRunner(args, cfgs)
        save_config(cfgs, agent_runner.runner.logdir)
        agent_runner.run()
    else:
        cfgs = parse_cfgs(args)
        agent_runner = AgentRunner(args, cfgs)
        agent_runner.run()

    import json
    with open(f"logs/{cfgs['CONTROLLER']['cav_manager']['dataset']}.json", 'w') as fh:
        json.dump(agent_runner.runner.forward_runner.shared_modules.spatial_alignment.gvr.data, fh)