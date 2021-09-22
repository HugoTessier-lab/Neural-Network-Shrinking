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


class MutableChannelMapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MutableChannelMapper, self).__init__()
        self.bias = None
        self.stride = 1
        self.padding = 0
        self.dilation = 1
        self.groups = 1

        self.weight = nn.Parameter(torch.zeros(len(out_channels), len(in_channels), 1, 1), requires_grad=True)
        self.weight.data[torch.nonzero(out_channels)[:, 0], torch.nonzero(in_channels)[:, 0], 0, 0] = 1

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ChannelMapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelMapper, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if 0 in x.shape:
            return x
        if x.shape[1] != len(self.in_channels):
            raise ValueError
        if x.shape[1] == len(self.in_channels) == len(self.out_channels):
            return x
        else:
            x = x[:, torch.nonzero(self.in_channels)[:, 0], :, :]
            new_x = torch.zeros(x.shape[0], self.out_channels.shape[0], x.shape[2], x.shape[3]).to(x.device)
            new_x[:, torch.nonzero(self.out_channels)[:, 0], :, :] = x
            return new_x

    def update(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels


def freeze_mapper(mutable):
    in_channels = torch.sum(mutable.weight.detach(), dim=(0, 2, 3))
    out_channels = torch.sum(mutable.weight.detach(), dim=(1, 2, 3))
    return ChannelMapper(in_channels, out_channels)


def unfreeze_mapper(mapper):
    return MutableChannelMapper(mapper.in_channels, mapper.out_channels)


def create_channel_mapper(channels):
    in_channels = torch.ones(channels)
    out_channels = torch.ones(channels)
    return ChannelMapper(in_channels, out_channels)
