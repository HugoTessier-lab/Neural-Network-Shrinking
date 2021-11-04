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


class FilterL1:
    def __init__(self):
        self.filters_count = None
        self.filters_l1 = None
        self.threshold = None
        self.total = None

    def init_model(self, model):
        count = []
        l1 = []
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                filters = [m.weight.data[i] for i in range(m.weight.shape[0])]
                filters_count = [len(i.flatten()) for i in filters]
                filters_l1 = [i.abs().sum() for i in filters]
                count.extend(filters_count)
                l1.extend(filters_l1)

        l1, indices = torch.sort(torch.Tensor(l1))
        count = torch.Tensor(count)[indices]
        self.filters_count = count
        self.filters_l1 = l1
        self.total = count.sum()

    def set_pruning_rate(self, rate):
        rate = self.total * rate
        count = 0
        index = 0
        while count + self.filters_count[index] < rate:
            count += self.filters_count
            index += 1
        self.threshold = self.filters_l1[index]

    def get_module_mask(self, mod):
        if isinstance(mod, torch.nn.Conv2d):
            filters = [mod.weight.data[i] for i in range(mod.weight.shape[0])]
            filters_ths = torch.Tensor([i.abs().sum() for i in filters]) > self.threshold
            mask = torch.ones(mod.weight.shape) * filters_ths[:, None, None, None]
            return mask.float()
        else:
            return torch.ones(mod.weight.shape).float()


class FilterL2:
    def __init__(self):
        self.filters_count = None
        self.filters_l2 = None
        self.threshold = None
        self.total = None

    def init_model(self, model):
        count = []
        l2 = []
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                filters = [m.weight.data[i] for i in range(m.weight.shape[0])]
                filters_count = [len(i.flatten()) for i in filters]
                filters_l2 = [i.pow(2).sum() for i in filters]
                count.extend(filters_count)
                l2.extend(filters_l2)

        l2, indices = torch.sort(torch.Tensor(l2))
        count = torch.Tensor(count)[indices]
        self.filters_count = count
        self.filters_l2 = l2
        self.total = count.sum()

    def set_pruning_rate(self, rate):
        rate = self.total * rate
        count = 0
        index = 0
        while count + self.filters_count[index] < rate:
            count += self.filters_count
            index += 1
        self.threshold = self.filters_l2[index]

    def get_module_mask(self, mod):
        if isinstance(mod, torch.nn.Conv2d):
            filters = [mod.weight.data[i] for i in range(mod.weight.shape[0])]
            filters_ths = torch.Tensor([i.pow(2).sum() for i in filters]) > self.threshold
            mask = torch.ones(mod.weight.shape) * filters_ths[:, None, None, None]
            return mask.float()
        else:
            return torch.ones(mod.weight.shape).float()
