from utils.pruning.pruning_criterion import GateMagnitude, generate_mask
import torch


def get_masked_params_count(network, pruner, threshold_generator, mask_rate):
    th = threshold_generator(mask_rate)
    mask = generate_mask(network, th)
    # t = time()
    pruner.apply_mask(mask)
    pruner.compute_pruning()
    # print('Apply mask ', time() - t)
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


if __name__ == '__main__':
    from models.hrnet import hrnet18
    from utils.pruning.pruner import Pruner
    from time import time
    from utils.pruning.custom_layers import Gate, ChannelMapper, MutableChannelMapper
    import numpy as np
    import random


    def set_seed(seed=0):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    set_seed(7)

    m = hrnet18()
    m.eval()

    for mod in m.modules():
        if isinstance(mod, Gate):
            mod.weight.data = torch.rand(mod.weight.data.shape).to(mod.weight.data.device)

    p = Pruner(m, (1, 3, 100, 100))

    begin = time()
    mask = find_mask(m, p, 0.95)
    print('Elapsed time ', time() - begin)
    p.apply_mask(mask)
    p.compute_pruning()

    # for name, mod in p.mock.named_modules():
        # if name == 'stage3.stage.1.branches.branches.0.3.downsample':
        #     print(name)
        #
        #     weight_mask = mod.weight.grad != 0
        #     print('weight_mask.int()', weight_mask.int())
        #     preserved_filters = weight_mask.float().mean(dim=(1, 2, 3)) != 0
        #     preserved_kernels = weight_mask.float().mean(dim=(0, 2, 3)) != 0
        #
        #     print('preserved_filters', preserved_filters)
        #     print('preserved_kernels', preserved_kernels)
        #
        #     weight = mod.weight.data * weight_mask
        #     weight = weight[preserved_filters]
        #     weight = weight[:, preserved_kernels]
        #
        #     in_channels = torch.sum(weight, dim=(0, 2, 3))
        #     out_channels = torch.sum(weight, dim=(1, 2, 3))
        #
        #     print('in_channels', in_channels)
        #     print('out_channels', out_channels)
        # if isinstance(mod, MutableChannelMapper):
        #     print(name)
        #     print((mod.weight.grad[:, :, 0, 0] != 0).int())
        #     print()

    before = len(torch.cat([i.flatten() for i in m.parameters()]))

    begin = time()
    m(torch.rand(1, 3, 512, 1024))
    forward1 = time() - begin
    print('Forward time before ', forward1)
    # torch.onnx.export(m, torch.rand(1, 3, 512, 1024), 'unpruned.onnx', verbose=True, opset_version=11,
    #                   do_constant_folding=True, export_params=True)

    p.shrink_model(m)
    after = len(torch.cat([i.flatten() for i in m.parameters()]))
    print('Params count before ', before, ', after ', after, ', ratio ', round(after / before, 2))
    begin = time()

    # for name, mod in m.named_modules():
    #     if isinstance(mod, ChannelMapper):
    #         print(name, mod.in_channels)

    m(torch.rand(1, 3, 512, 1024))
    forward2 = time() - begin
    print('Froward time after ', forward2)
    # torch.onnx.export(m, torch.rand(1, 3, 512, 1024), 'pruned.onnx', verbose=True, opset_version=11,
    #                   do_constant_folding=True, export_params=True)

    print('Speedup : ', round(forward1 / forward2, 2))

# seed 2 : stage3.stage.1.branches.branches.0.3.downsample
# torch.Size([1, 1, 128, 256]) torch.Size([0]) Parameter containing:
# tensor([0.8597], requires_grad=True) tensor([0., 1.]) tensor([1.])

# seed 7 : torch.Size([1, 2, 128, 256]) torch.Size([0]) Parameter containing:
# tensor([0.9163, 0.8619], requires_grad=True) tensor([1., 0., 0., 1.]) tensor([1., 1.])