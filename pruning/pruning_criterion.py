import torch
from pruning.custom_layers import Gate


class GateMagnitude:
    def __init__(self, model):
        self.params = torch.cat([i.weight.data.flatten().abs()
                                 for i in model.modules()
                                 if isinstance(i, Gate)]).sort()[0]

    def __call__(self, i):
        return self.params[i]

    def __len__(self):
        return len(self.params)


def generate_mask(network, threshold):
    mask = []
    for m in network.modules():
        if isinstance(m, Gate):
            mask.append((m.weight.data.abs() >= threshold).float())
    return mask
