import math

import torch.nn

from shrinking.pruner import Pruner
from pruning.target_calculation import find_mask


class SWD:
    def __init__(self, model, dataset, epochs, a_min, a_max, pruning_rate, image_shape, weight_decay,
                 pruning_criterion):
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
        self.pruning_criterion = pruning_criterion

    def get_a(self):
        return self.a_min * (math.pow(self.a_max / self.a_min, self.current_step / self.total_steps))

    def step(self):
        if self.step_in_epoch == self.dataset_length:
            self.step_in_epoch = 0
        if self.step_in_epoch == 0:
            self.mask = find_mask(self.model, self.pruner, self.pruning_rate, self.pruning_criterion)
        self.current_step += 1
        self.step_in_epoch += 1
        a = self.get_a()
        i = 0
        for m in self.model.modules():
            if hasattr(m, 'weight'):
                m.weight.grad += a * self.weight_decay * (m.weight.data * (self.mask[i] == 0))
                i += 1
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.grad += a * self.weight_decay * (m.bias.data * (self.mask[i] == 0))
                    i += 1

    def prune(self):
        self.mask = find_mask(self.model, self.pruner, self.pruning_rate, self.pruning_criterion)
        self.pruner.apply_mask(self.mask)
        self.pruner.shrink_model(self.model)

    def get_name(self, mode):
        return f'_swd_pruning_rate_{self.pruning_rate}_amin_{self.a_min}_amax_{self.a_max}'


class Liu2017:
    def __init__(self, model, pruning_criterion, pruning_rate, penalty, image_shape):
        self.model = model
        device = None
        for p in model.parameters():
            device = p.device
            break
        self.pruner = Pruner(model, image_shape, device)
        self.pruning_rate = pruning_rate
        self.pruning_criterion = pruning_criterion
        self.penalty = penalty

    def _smooth_l1(self, t):
        return ((t >= 1).float() - (t <= -1).float() + (t * (t.abs() < 1))) * self.penalty

    def step(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.grad += self._smooth_l1(m.weight.data)

    def prune(self):
        mask = find_mask(self.model, self.pruner, self.pruning_rate, self.pruning_criterion)
        self.pruner.apply_mask(mask)
        self.pruner.shrink_model(self.model)

    def get_name(self, mode):
        if mode == 'base':
            return f'_liu2017_penalty_{self.penalty}'
        elif mode == 'pruned':
            return f'_liu2017_penalty_{self.penalty}_pruning_rate_{self.pruning_rate}'
        else:
            raise ValueError


def get_pruning_method(args, model, dataset, pruning_criterion):
    if args.pruning_method == 'none':
        return None
    elif args.pruning_method == 'swd':
        return SWD(model, dataset, args.epochs, args.a_min, args.a_max, args.pruning_rate,
                   args.pruner_image_shape, args.wd, pruning_criterion)
    elif args.pruning_method == 'liu2017':
        return Liu2017(model, pruning_criterion, args.pruning_rate, args.liu2017_penalty, args.pruner_image_shape)
    else:
        print('ERROR : non existing pruning method type.')
        raise ValueError
