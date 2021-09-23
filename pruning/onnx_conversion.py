import torch


def to_onnx(model, path, image_shape, device):
    model(torch.rand(image_shape).to(device))
    torch.onnx.export(m, torch.rand(image_shape).to(device), path, verbose=True, opset_version=11,
                      do_constant_folding=True, export_params=True)


if __name__ == '__main__':
    from semantic_segmentation.models.hrnet import hrnet18
    from pruning.pruner import Pruner
    from time import time
    from pruning.custom_layers import Gate
    import numpy as np
    import random
    from pruning.target_calculation import find_mask


    def set_seed(seed=0):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    set_seed(0)

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

    before = len(torch.cat([i.flatten() for i in m.parameters()]))

    begin = time()
    m(torch.rand(1, 3, 100, 100))
    forward1 = time() - begin
    print('Forward time before ', forward1)
    to_onnx(m, 'unpruned.onnx', (1, 3, 100, 100), 'cpu')

    p.shrink_model(m)
    after = len(torch.cat([i.flatten() for i in m.parameters()]))
    print('Params count before ', before, ', after ', after, ', ratio ', round(after / before, 2))
    begin = time()

    m(torch.rand(1, 3, 100, 100))
    forward2 = time() - begin
    print('Froward time after ', forward2)
    print('Speedup : ', round(forward1 / forward2, 2))

    # to_onnx(m, 'pruned.onnx', (1, 3, 100, 100), 'cpu')
