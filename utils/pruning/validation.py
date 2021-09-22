from utils.pruning.pruner import Pruner
from models import hrnet
from utils.pruning.custom_layers import Gate
import torch
import numpy as np
import random


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def constant_rate_mask(model, rate):
    masks = []
    for m in model.modules():
        if isinstance(m, Gate):
            mask = np.ones(m.weight.shape)
            mask[:int(len(mask) * rate)] = 0
            np.random.shuffle(mask)
            masks.append(torch.Tensor(mask))
    return masks


def constant_rate_test(rate):
    print('Rate : ', round(rate, 1))
    model = hrnet.hrnet18(False)
    model(torch.rand((1, 3, 100, 100)))
    pruner = Pruner(model, (1, 3, 100, 100))
    mask = constant_rate_mask(model, rate)
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)))


def random_rate_mask(model):
    masks = []
    for m in model.modules():
        if isinstance(m, Gate):
            rate = torch.rand(1).item()
            mask = np.ones(m.weight.shape)
            mask[:int(len(mask) * rate)] = 0
            np.random.shuffle(mask)
            masks.append(torch.Tensor(mask))
    return masks


def random_rate_test(seed):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False)
    model(torch.rand((1, 3, 100, 100)))
    pruner = Pruner(model, (1, 3, 100, 100))
    mask = random_rate_mask(model)
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)))


def random_cuts(model):
    masks = []
    for m in model.modules():
        if isinstance(m, Gate):
            draw = torch.rand(1).item()
            mask = np.ones(m.weight.shape)
            if draw > 0.97:
                mask.fill(0)
            masks.append(torch.Tensor(mask))
    return masks


def random_cuts_test(seed):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False)
    model(torch.rand((1, 3, 100, 100)))
    pruner = Pruner(model, (1, 3, 100, 100))
    mask = random_cuts(model)
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)))


def random_cuts_and_rates(model):
    masks = []
    for m in model.modules():
        if isinstance(m, Gate):
            draw = torch.rand(1).item()
            mask = np.ones(m.weight.shape)
            if draw > 0.97:
                mask.fill(0)
            else:
                rate = torch.rand(1).item()
                mask = np.ones(m.weight.shape)
                mask[:int(len(mask) * rate)] = 0
                np.random.shuffle(mask)
            masks.append(torch.Tensor(mask))
    return masks


def random_cuts_and_rates_test(seed):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False)
    model(torch.rand((1, 3, 100, 100)))
    pruner = Pruner(model, (1, 3, 100, 100))
    mask = random_cuts_and_rates(model)
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)))


if __name__ == '__main__':
    print('Same rate on all gates\n')
    rates = np.linspace(0, 1, 11)
    for r in rates:
        constant_rate_test(r)

    print('\nRandom rate on each gates (uniformly distributed)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        random_rate_test(s)

    print('\nRandomly cut gates (probability 3%)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        random_cuts_test(s)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each gates\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        random_cuts_and_rates_test(s)
