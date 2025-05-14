# This module provides prototypes for CAVs/agents.
# The prototype has the following features:
# 1. Data processing logics for each prototyped agent/CAV.
# 2. All intermediate data processed are stored locally at prototype class.
# 3. Specify the requesting and responding CPMs

import importlib


def get_prototype(module_full_path: str):
    module_name, cls_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'cosense3d.agents.cav_prototype.{module_name}')
    cls_obj = getattr(module, cls_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    return cls_obj