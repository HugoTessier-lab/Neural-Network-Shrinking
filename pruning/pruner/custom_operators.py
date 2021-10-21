import torch
from torch import nn as nn
from torch.nn import functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor([]), requires_grad=False)

    def forward(self, x):
        return torch.Tensor([]).to(x.device)


class Gate(nn.Module):
    def __init__(self, channels):
        super(Gate, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels), requires_grad=True)

    def forward(self, x):
        result = x * self.weight[None, :, None, None]
        return result


def add(a, b):
    if 0 not in a.size() and 0 in b.size():
        return a
    elif 0 in a.size() and 0 not in b.size():
        return b
    elif 0 not in a.size() and 0 not in b.size():
        return a + b
    else:
        return torch.Tensor([]).to(a.device)


class Adder(nn.Module):
    def __init__(self, channels):
        super(Adder, self).__init__()

        self.in_channels_a = torch.linspace(0, channels - 1, channels).long()
        self.out_channels_a = torch.linspace(0, channels - 1, channels).long()

        self.in_channels_b = torch.linspace(0, channels - 1, channels).long()
        self.out_channels_b = torch.linspace(0, channels - 1, channels).long()

        self.channels = channels

    @staticmethod
    def single_input(i, in_channels, out_channels, channels):
        if not (len(in_channels) == i.shape[1] and max(in_channels) == i.shape[1]):
            i = i[:, in_channels, :, :]
        if channels == i.shape[1]:
            return i
        else:
            dims = i.shape
            device = i.device
            new = torch.zeros(dims[0], channels, dims[2], dims[3]).to(device)
            new[:, out_channels, :, :] = i
            return new

    def forward(self, x, shortcut_input):
        if 0 in x.shape and 0 in shortcut_input.shape:
            return torch.Tensor([]).to(x.device)
        if 0 not in x.shape and 0 in shortcut_input.shape:
            return self.single_input(x, self.in_channels_a, self.out_channels_a, self.channels)
        if 0 in x.shape and 0 not in shortcut_input.shape:
            return self.single_input(shortcut_input, self.in_channels_b, self.out_channels_b, self.channels)

        if not (len(self.in_channels_a) == x.shape[1] and max(self.in_channels_a) == len(self.in_channels_a)):
            x = x[:, self.in_channels_a, :, :]
        if not (len(self.in_channels_b) == shortcut_input.shape[1]
                and max(self.in_channels_b) == len(self.in_channels_b)):
            shortcut_input = shortcut_input[:, self.in_channels_b, :, :]

        if x.shape[1] == self.channels and shortcut_input.shape[1] == self.channels:
            return x + shortcut_input
        elif x.shape[1] != self.channels and shortcut_input.shape[1] == self.channels:
            shortcut_input[:, self.out_channels_a, :, :].add_(x)
            return shortcut_input
        elif x.shape[1] == self.channels and shortcut_input.shape[1] != self.channels:
            x[:, self.out_channels_b, :, :].add_(shortcut_input)
            return x
        else:
            dims = x.shape
            device = x.device
            new = torch.zeros(dims[0], self.channels, dims[2], dims[3]).to(device)
            new[:, self.out_channels_a, :, :].add_( x[:, self.in_channels_a, :, :])
            new[:, self.out_channels_b, :, :].add_(shortcut_input[:, self.in_channels_b, :, :])
            return new

    def update(self, in_channels_a, out_channels_a, in_channels_b, out_channels_b, channels_number):
        self.in_channels_a = in_channels_a
        self.out_channels_a = out_channels_a
        self.in_channels_b = in_channels_b
        self.out_channels_b = out_channels_b
        self.channels = channels_number


class MutableAdder(nn.Module):
    class MappingConvolution(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(MutableAdder.MappingConvolution, self).__init__()
            self.weight = nn.Parameter(torch.zeros(len(out_channels), len(in_channels), 1, 1), requires_grad=True)
            self.weight.data[out_channels, in_channels, 0, 0] = 1

            self.bias = None
            self.stride = 1
            self.padding = 0
            self.dilation = 1
            self.groups = 1

        def forward(self, x):
            return F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)

    def __init__(self, in_channels_a, out_channels_a, in_channels_b, out_channels_b):
        super(MutableAdder, self).__init__()

        self.conv1 = MutableAdder.MappingConvolution(in_channels_a, out_channels_a)
        self.conv2 = MutableAdder.MappingConvolution(in_channels_b, out_channels_b)

    def forward(self, x, shortcut_input):
        x = self.conv1(x)
        shortcut_input = self.conv2(shortcut_input)
        return x + shortcut_input


def unfreeze_adder(adder):
    return MutableAdder(adder.in_channels_a, adder.out_channels_a, adder.in_channels_b, adder.out_channels_b)
