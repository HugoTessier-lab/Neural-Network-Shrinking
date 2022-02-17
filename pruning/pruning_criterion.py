import torch
import torch.nn as nn


class BatchNormGlobal:
    def __init__(self):
        self.params = None
        self.threshold = None

    def init_model(self, model):
        self.params = torch.cat([i.weight.data.flatten().abs()
                                 for i in model.modules()
                                 if isinstance(i, nn.BatchNorm2d)]).sort()[0]

    def set_pruning_rate(self, rate):
        self.threshold = self.params[int(rate * len(self.params))]

    def get_module_mask(self, mod):
        if isinstance(mod, nn.BatchNorm2d):
            mask = (mod.weight.data.abs() >= self.threshold).float()
            return [mask, mask]
        else:
            if hasattr(mod, 'bias'):
                if mod.bias is not None:
                    return [torch.ones(mod.weight.shape).to(mod.weight.device),
                            torch.ones(mod.bias.shape).to(mod.bias.device)]
            return [(mod.weight.data.abs() >= 0).float()]
