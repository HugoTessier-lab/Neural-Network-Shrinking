import torch
from pruning.pruner.custom_operators import Gate


class GateGlobalMagnitude:
    def __init__(self):
        self.params = None
        self.threshold = None

    def init_model(self, model):
        self.params = torch.cat([i.weight.data.flatten().abs()
                                 for i in model.modules()
                                 if isinstance(i, Gate)]).sort()[0]

    def set_pruning_rate(self, rate):
        self.threshold = self.params[int(rate * len(self.params))]

    def get_module_mask(self, mod):
        if isinstance(mod, Gate):
            return (mod.weight.data.abs() >= self.threshold).float()
        else:
            return (mod.weight.data.abs() >= 0).float()
