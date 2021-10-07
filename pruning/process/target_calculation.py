import torch


def generate_mask(network, pruning_criterion):
    mask = []
    for mod in network.modules():
        if hasattr(mod, 'weight'):
            mask.append(pruning_criterion.get_module_mask(mod))
    return mask


def get_masked_params_count(network, pruner, pruning_criterion):
    mask = generate_mask(network, pruning_criterion)
    pruner.apply_mask(mask)
    return pruner.get_params_count()


def find_mask(network, pruner, target, pruning_criterion):
    pruning_criterion.init_model(network)
    params_total = len(torch.cat([i.flatten() for i in network.parameters()]))

    lower_bound = 0
    upper_bound = 1
    mask_rate = 0.5

    previous_count = 0

    iterations = 0
    while True:
        iterations += 1
        pruning_criterion.set_pruning_rate(mask_rate)
        pruning_count = get_masked_params_count(network, pruner, pruning_criterion).item()
        pruning_rate = 1 - (pruning_count / params_total)
        print(iterations, lower_bound, upper_bound, mask_rate, previous_count, pruning_count)
        if pruning_rate > target:
            upper_bound = mask_rate
            mask_rate = (lower_bound + mask_rate) / 2
        elif pruning_rate < target:
            lower_bound = mask_rate
            mask_rate = (upper_bound + mask_rate) / 2
        if pruning_rate == target or previous_count == pruning_count:
            pruning_criterion.set_pruning_rate(mask_rate)
            return generate_mask(network, pruning_criterion)
        previous_count = pruning_count


if __name__ == '__main__':
    from semantic_segmentation.networks import hrnet
    from pruning.process.pruning_criterion import GateGlobalMagnitude
    from pruning.pruner.pruner import Pruner
    from pruning.pruner.custom_operators import Gate
    import time


    def set_seed(seed=0):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


    set_seed(0)
    m = hrnet.hrnet18(gates=True, mappers=True)
    for mod in m.modules():
        if isinstance(mod, Gate):
            mod.weight.data = torch.rand(mod.weight.data.shape)
    p = Pruner(m, (1, 3, 100, 100), 'cpu')

    crit = GateGlobalMagnitude()
    t = 0.5

    begin = time.time()
    ma = find_mask(m, p, t, crit)
    print('Elapsed time : ', round(time.time() - begin, 2))

    p.apply_mask(ma)
    count1 = len(torch.cat([i.flatten() for i in m.parameters()]))
    p.shrink_model(m)

    count2 = len(torch.cat([i.flatten() for i in m.parameters()]))

    print(f'Before {count1}, after {count2}, ratio {round(count2 / count1, 2)}')
