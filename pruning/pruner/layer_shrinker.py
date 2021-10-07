import torch


def shrink_conv(conv, mock):
    weight_mask = (mock.weight.data != 0) & (mock.weight.grad != 0)
    conv.weight.data = conv.weight.data * weight_mask
    preserved_filters = (conv.weight.data.abs().mean(dim=(1, 2, 3)) != 0)
    preserved_kernels = conv.weight.data.abs().mean(dim=(0, 2, 3)) != 0

    if True not in preserved_kernels:
        preserved_filters.fill_(False)
    conv.weight.data = conv.weight.data[preserved_filters]
    conv.weight.data = conv.weight.data[:, preserved_kernels]
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[preserved_filters]


def shrink_bn(bn, mock):
    weight_mask = (mock.weight.data != 0) & (mock.weight.grad != 0)
    bn.weight.data = bn.weight.data * weight_mask
    bn.bias.data = bn.bias.data * weight_mask
    preserved_filters = bn.weight.data.abs() != 0

    bn.weight.data = bn.weight.data[preserved_filters]
    bn.bias.data = bn.bias.data[preserved_filters]
    bn.running_mean = bn.running_mean[preserved_filters]
    bn.running_var = bn.running_var[preserved_filters]


def shrink_gate(gate, mock):
    weight_mask = (mock.weight.data != 0) & (mock.weight.grad != 0)
    gate.weight.data = gate.weight.data * weight_mask
    preserved_filters = (gate.weight.data.abs() != 0)
    gate.weight.data = gate.weight.data[preserved_filters]


def shrink_identity(index, mock):
    weight_mask = mock.weight.grad != 0
    preserved_filters = weight_mask.float().mean(dim=(1, 2, 3)) != 0
    preserved_kernels = weight_mask.float().mean(dim=(0, 2, 3)) != 0

    weight = mock.weight.data * weight_mask
    weight = weight[preserved_filters]
    weight = weight[:, preserved_kernels]

    in_channels = torch.sum(weight, dim=(0, 2, 3))
    out_channels = torch.sum(weight, dim=(1, 2, 3))

    in_channels = torch.nonzero(in_channels)[:, 0]
    out_channels = torch.nonzero(out_channels)[:, 0]
    index.update(in_channels, out_channels)
