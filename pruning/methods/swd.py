import math
from pruning.pruner.pruner import Pruner
from pruning.process.pruning_criterion import BatchNormGlobal
from pruning.process.target_calculation import find_mask


class SWD:
    def __init__(self, model, dataset, epochs, a_min, a_max, pruning_rate, image_shape, weight_decay):
        self.model = model
        self.total_steps = len(dataset) * epochs
        self.dataset_length = len(dataset)
        self.a_min = a_min
        self.a_max = a_max
        self.current_step = 0
        device = None
        for p in model.parameters():
            device = p.device
            break
        self.pruner = Pruner(model, image_shape, device)
        self.pruning_rate = pruning_rate

        self.step_in_epoch = 0
        self.mask = None
        self.weight_decay = weight_decay

    def get_a(self):
        return self.a_min * (math.pow(self.a_max / self.a_min, self.current_step / self.total_steps))

    def step(self):
        if self.step_in_epoch == self.dataset_length:
            self.step_in_epoch = 0
        if self.step_in_epoch == 0:
            self.mask = find_mask(self.model, self.pruner, self.pruning_rate, BatchNormGlobal())
        self.current_step += 1
        self.step_in_epoch += 1
        a = self.get_a()
        i = 0
        for m in self.model.modules():
            if hasattr(m, 'weight'):
                m.weight.grad += a * self.weight_decay * (m.weight.grad * (self.mask[i] != 0))
                i += 1
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.grad += a * self.weight_decay * (m.bias.grad * (self.mask[i] != 0))
                    i += 1

    def prune(self):
        self.pruner.shrink_model(self.model)
