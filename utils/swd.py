import math
import torch


class SWD:
    def __init__(self, dataset_length, max_epochs, a_min, a_max, pruning_rate):
        self.total_steps = dataset_length * max_epochs
        self.a_min = a_min
        self.a_max = a_max
        self.current_step = 0
        self.pruning_rate = pruning_rate

    def get_a(self):
        return self.a_min * (math.pow(self.a_max / self.a_min, self.current_step / self.total_steps))

    def __call__(self, model):
        params = torch.cat([i.weight.data.abs().flatten()
                            for i in model.modules() if isinstance(i, torch.nn.Conv2d)]).sort()[0]
        return params[:int(len(params) * self.pruning_rate)].pow(2).sum() * self.get_a()

    def prune(self, model):
        params = torch.cat([i.weight.data.abs().flatten()
                            for i in model.modules() if isinstance(i, torch.nn.Conv2d)]).sort()[0]
        ths = params[int(len(params) * self.pruning_rate)]
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                m.weight.data = m.weight.data * (m.weight.data.abs() > ths).float()
        return model
