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

    @staticmethod
    def name():
        return 'BatchNormGlobal'


class BatchNormLocal:
    def __init__(self):
        self.rate = None

    def init_model(self, model):
        pass

    def set_pruning_rate(self, rate):
        self.rate = rate

    def get_module_mask(self, mod):
        if isinstance(mod, nn.BatchNorm2d):
            weights = mod.weight.data.abs().sort()[0]
            threshold = weights[int(self.rate * len(weights))]
            mask = (mod.weight.data.abs() >= threshold).float()
            return [mask, mask]
        else:
            if hasattr(mod, 'bias'):
                if mod.bias is not None:
                    return [torch.ones(mod.weight.shape).to(mod.weight.device),
                            torch.ones(mod.bias.shape).to(mod.bias.device)]
            return [(mod.weight.data.abs() >= 0).float()]

    @staticmethod
    def name():
        return 'BatchNormLocal'


class BatchNormSafe:
    def __init__(self):
        self.params = None
        self.threshold = None
        self.layers = None

    def init_model(self, model):
        self.layers = model.get_safe_layers()
        self.params = torch.cat([i.weight.data.flatten().abs()
                                 for i in self.layers
                                 if isinstance(i, nn.BatchNorm2d)]).sort()[0]

    def set_pruning_rate(self, rate):
        self.threshold = self.params[int(rate * len(self.params))]

    def get_module_mask(self, mod):
        if isinstance(mod, nn.BatchNorm2d) and mod in self.layers:
            mask = (mod.weight.data.abs() >= self.threshold).float()
            return [mask, mask]
        else:
            if hasattr(mod, 'bias'):
                if mod.bias is not None:
                    return [torch.ones(mod.weight.shape).to(mod.weight.device),
                            torch.ones(mod.bias.shape).to(mod.bias.device)]
            return [(mod.weight.data.abs() >= 0).float()]

    @staticmethod
    def name():
        return 'BatchNormSafe'


def get_pruning_criterion(name):
    if name == 'global':
        return BatchNormGlobal()
    elif name == 'local':
        return BatchNormLocal()
    elif name == 'safe':
        return BatchNormSafe()
    else:
        print('ERROR : non existing pruning criterion type.')
        raise ValueError
