from cosense3d.dataset.pipeline.loading import *
from cosense3d.dataset.pipeline.transform import *


class Pipeline(object):
    """Composes several processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, cfgs):
        self.processes = []
        if isinstance(cfgs, list):
            for cfg in cfgs:
                for k, v in cfg.items():
                    self.build_process(k, v)
        elif isinstance(cfgs, OrderedDict):
            for k, v in cfgs.items():
                self.build_process(k, v)
        else:
            raise NotImplementedError

    def build_process(self, k, v):
        cls = globals().get(k, None)
        assert cls is not None, f"Pipeline process node {k} not found."
        self.processes.append(cls(**v))

    def __call__(self, data_dict):
        for p in self.processes:
            p(data_dict)
        return data_dict


