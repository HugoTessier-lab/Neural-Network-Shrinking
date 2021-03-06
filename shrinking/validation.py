from shrinking.pruner import Pruner
from utils.networks import hrnet, resnet
import torch
import torch.nn as nn
import numpy as np
import random
from time import time

errors = 0


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def test(mask_method, model, size, device, dtype):
    input_image = torch.rand(size).to(device)
    model(input_image)
    pruner = Pruner(model, size, device, dtype=dtype, remove_biases=True)
    mask = mask_method(model, device)
    pruner.apply_mask(mask)
    unpruned_parameters_count = len(torch.cat([i.flatten() for i in model.parameters()]))
    print('\tUnpruned parameters count : ', unpruned_parameters_count)
    predicted_pruned_parameters_count = pruner.get_params_count().item()
    print('\tPredicted pruned parameters count : ', predicted_pruned_parameters_count)
    pruner.shrink_model(model)
    pruned_parameters_count = len(torch.cat([i.flatten() for i in model.parameters()]))
    print('\tPruned parameters count : ', pruned_parameters_count)
    if predicted_pruned_parameters_count != pruned_parameters_count:
        global errors
        errors += 1
        print('\t\tWARNING : discrepancy in prediction !')
        print('\t\tDifference between shrunken and predicted : ',
              pruned_parameters_count - predicted_pruned_parameters_count)
        print('\t\tshrunken/predicted ratio ', round(pruned_parameters_count / predicted_pruned_parameters_count, 2))
    model(input_image)
    model.freeze(size)
    model(input_image)


def constant_rate_mask(rate):
    print('Rate : ', round(rate, 1))

    def masker(model, device):
        masks = []
        for m in model.modules():
            if hasattr(m, 'weight'):
                if isinstance(m, nn.BatchNorm2d):
                    mask = np.ones(m.weight.shape)
                    mask[:int(len(mask) * rate)] = 0
                    np.random.shuffle(mask)
                    masks.append(torch.Tensor(mask).to(device))
                else:
                    masks.append(torch.ones(m.weight.shape).to(device))
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    masks.append(torch.ones(m.bias.shape).to(device))
        return masks

    return masker


def random_rate_mask(seed):
    set_seed(seed)
    print('Seed : ', seed)

    def masker(model, device):
        masks = []
        for m in model.modules():
            if hasattr(m, 'weight'):
                if isinstance(m, nn.BatchNorm2d):
                    rate = torch.rand(1).item()
                    mask = np.ones(m.weight.shape)
                    mask[:int(len(mask) * rate)] = 0
                    np.random.shuffle(mask)
                    masks.append(torch.Tensor(mask).to(device))
                else:
                    masks.append(torch.ones(m.weight.shape).to(device))
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    masks.append(torch.ones(m.bias.shape).to(device))
        return masks

    return masker


def random_cuts(seed):
    set_seed(seed)
    print('Seed : ', seed)

    def masker(model, device):
        masks = []
        for m in model.modules():
            if hasattr(m, 'weight'):
                if isinstance(m, nn.BatchNorm2d):
                    draw = torch.rand(1).item()
                    mask = np.ones(m.weight.shape)
                    if draw > 0.97:
                        mask.fill(0)
                    masks.append(torch.Tensor(mask).to(device))
                else:
                    masks.append(torch.ones(m.weight.shape).to(device))
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    masks.append(torch.ones(m.bias.shape).to(device))
        return masks

    return masker


def random_cuts_and_rates(seed):
    set_seed(seed)
    print('Seed : ', seed)

    def masker(model, device):
        masks = []
        for m in model.modules():
            if hasattr(m, 'weight'):
                if isinstance(m, nn.BatchNorm2d):
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
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    masks.append(torch.ones(m.bias.shape).to(device))
        return masks

    return masker


def random_cuts_and_rates_filters(seed):
    set_seed(seed)
    print('Seed : ', seed)

    def masker(model, device):
        masks = []
        for n, m in model.named_modules():
            if hasattr(m, 'weight'):
                if isinstance(m, torch.nn.Conv2d):
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
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    masks.append(torch.ones(m.bias.shape).to(device))
        return masks

    return masker


def test_hrnet():
    input_size = (1, 3, 64, 64)
    print('On HRNet-18\n')
    print('Same rate on all BNs\n')
    rates = np.linspace(0, 1, 11)
    for r in rates:
        test(constant_rate_mask(r), hrnet.hrnet18(False, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom rate on each BN (uniformly distributed)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        test(random_rate_mask(s), hrnet.hrnet18(False, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandomly cut BNs (probability 3%)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        test(random_cuts(s), hrnet.hrnet18(False, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each BN\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        test(random_cuts_and_rates(s), hrnet.hrnet18(False, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each convolutional layer\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        test(random_cuts_and_rates_filters(s), hrnet.hrnet18(False, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)


def test_resnet():
    input_size = (1, 3, 32, 32)
    print('On ResNet-18\n')
    print('Same rate on all BN\n')
    rates = np.linspace(0, 1, 11)
    for r in rates:
        test(constant_rate_mask(r), resnet.resnet18(num_classes=10, in_planes=64, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom rate on each BN (uniformly distributed)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        test(random_rate_mask(s), resnet.resnet18(num_classes=10, in_planes=64, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandomly cut BN (probability 3%)\n')
    seeds = [i for i in range(10)]
    for s in seeds:
        test(random_cuts(s), resnet.resnet18(num_classes=10, in_planes=64, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each BN\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        test(random_cuts_and_rates(s),
             resnet.resnet18(num_classes=10, in_planes=64, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)

    print('\nRandom cut (probability 3%) or random rate (uniformly distributed) on each convolutional layer\n')
    seeds = [i for i in range(0, 10)]
    for s in seeds:
        test(random_cuts_and_rates_filters(s), resnet.resnet18(num_classes=10, in_planes=64, adder=True).to('cpu'),
             input_size, 'cpu', torch.float32)


if __name__ == '__main__':
    test_hrnet()
    test_resnet()
    print('Reported errors : ', errors)
