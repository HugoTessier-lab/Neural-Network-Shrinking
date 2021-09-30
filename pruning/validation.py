from pruning.pruner.pruner import Pruner
from semantic_segmentation.networks import hrnet
from pruning.pruner.custom_layers import Gate
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


def constant_rate_mask(model, rate, device):
    masks = []
    for m in model.modules():
        if hasattr(m, 'weight'):
            if isinstance(m, Gate):
                mask = np.ones(m.weight.shape)
                mask[:int(len(mask) * rate)] = 0
                np.random.shuffle(mask)
                masks.append(torch.Tensor(mask).to(device))
            else:
                masks.append(torch.ones(m.weight.shape).to(device))
    return masks


def constant_rate_test(rate, device, dtype):
    print('Rate : ', round(rate, 1))
    model = hrnet.hrnet18(False, gates=True, mappers=True).to(device)
    model(torch.rand((1, 3, 100, 100)).to(device))
    pruner = Pruner(model, (1, 3, 100, 100), device, dtype=dtype)
    mask = constant_rate_mask(model, rate, device)
    pruner.apply_mask(mask)
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)).to(device))


def random_rate_mask(model, device):
    masks = []
    for m in model.modules():
        if hasattr(m, 'weight'):
            if isinstance(m, Gate):
                rate = torch.rand(1).item()
                mask = np.ones(m.weight.shape)
                mask[:int(len(mask) * rate)] = 0
                np.random.shuffle(mask)
                masks.append(torch.Tensor(mask).to(device))
            else:
                masks.append(torch.ones(m.weight.shape).to(device))
    return masks


def random_rate_test(seed, device, dtype):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False, gates=True, mappers=True).to(device)
    model(torch.rand((1, 3, 100, 100)).to(device))
    pruner = Pruner(model, (1, 3, 100, 100), device, dtype=dtype)
    mask = random_rate_mask(model, device)
    pruner.apply_mask(mask)
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)).to(device))


def random_cuts(model, device):
    masks = []
    for m in model.modules():
        if hasattr(m, 'weight'):
            if isinstance(m, Gate):
                draw = torch.rand(1).item()
                mask = np.ones(m.weight.shape)
                if draw > 0.97:
                    mask.fill(0)
                masks.append(torch.Tensor(mask).to(device))
            else:
                masks.append(torch.ones(m.weight.shape).to(device))
    return masks


def random_cuts_test(seed, device, dtype):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False, gates=True, mappers=True).to(device)
    model(torch.rand((1, 3, 100, 100)).to(device))
    pruner = Pruner(model, (1, 3, 100, 100), device, dtype=dtype)
    mask = random_cuts(model, device)
    pruner.apply_mask(mask)
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)).to(device))


def random_cuts_and_rates(model, device):
    masks = []
    for m in model.modules():
        if hasattr(m, 'weight'):
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
                masks.append(torch.Tensor(mask).to(device))
            else:
                masks.append(torch.ones(m.weight.shape).to(device))
    return masks


def random_cuts_and_rates_test(seed, device, dtype):
    set_seed(seed)
    print('Seed : ', seed)
    model = hrnet.hrnet18(False, gates=True, mappers=True).to(device)
    model(torch.rand((1, 3, 100, 100)).to(device))
    pruner = Pruner(model, (1, 3, 100, 100), device, dtype=dtype)
    mask = random_cuts_and_rates(model, device)
    pruner.apply_mask(mask)
    print('\tUnpruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    print('\tPredicted pruned parameters count : ', pruner.get_params_count().item())
    pruner.shrink_model(model)
    print('\tPruned parameters count : ', len(torch.cat([i.flatten() for i in model.parameters()])))
    model(torch.rand((1, 3, 100, 100)).to(device))


if __name__ == '__main__':
    dev = 'cuda'
    ty = torch.float32
    print('Same rate on all gates\n')
    rates = np.linspace(0, 1, 11)
    for r in rates:
        constant_rate_test(r, dev, ty)

    print('\nRandom rate on each gates (uniformly distributed)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        random_rate_test(s, dev, ty)

    print('\nRandomly cut gates (probability 3%)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        random_cuts_test(s, dev, ty)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each gates\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        random_cuts_and_rates_test(s, dev, ty)
