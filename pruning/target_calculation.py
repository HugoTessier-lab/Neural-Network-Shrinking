from pruning.pruning_criterion import GateMagnitude, generate_mask
import torch


def get_masked_params_count(network, pruner, threshold_generator, mask_rate):
    th = threshold_generator(mask_rate)
    mask = generate_mask(network, th)
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    return pruner.get_params_count()


def find_mask(network, pruner, target):
    threshold_generator = GateMagnitude(network)
    params_total = len(torch.cat([i.flatten() for i in network.parameters()]))

    lower_bound = 0
    upper_bound = len(threshold_generator) - 1
    mask_rate = upper_bound // 2

    while True:
        pruning_count = get_masked_params_count(network, pruner, threshold_generator, mask_rate).item()
        pruning_rate = 1 - (pruning_count / params_total)
        if pruning_rate > target:
            upper_bound = mask_rate
            mask_rate = (lower_bound + mask_rate) // 2
        elif pruning_rate < target:
            lower_bound = mask_rate
            mask_rate = (upper_bound + mask_rate) // 2
        if pruning_rate == target or abs(lower_bound - upper_bound) <= 1:
            return generate_mask(network, threshold_generator(mask_rate))
