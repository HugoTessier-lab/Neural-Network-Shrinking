import torch


def generate_mask(network, pruning_criterion):
    mask = []
    for mod in network.modules():
        if hasattr(mod, 'weight'):
            mask.extend(pruning_criterion.get_module_mask(mod))
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
