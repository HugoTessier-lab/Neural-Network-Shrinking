import torch
from torch import nn as nn
from torch.nn import functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor([]), requires_grad=False)

    def forward(self, x, *args):
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

        self.in_channels_a = torch.linspace(0, channels - 1, channels, dtype=torch.int64)
        self.out_channels_a = torch.linspace(0, channels - 1, channels, dtype=torch.int64)

        self.in_channels_b = torch.linspace(0, channels - 1, channels, dtype=torch.int64)
        self.out_channels_b = torch.linspace(0, channels - 1, channels, dtype=torch.int64)

        self.channels = channels
        self.input_a_shape = None
        self.input_b_shape = None
        self.output_image_shape = None
        self.device = None

    @staticmethod
    def gather_channels(t, in_channels):
        if torch.equal(in_channels, torch.linspace(0, t.shape[1] - 1, t.shape[1], dtype=torch.int64)):
            return t
        shape = t.shape
        index = in_channels.view(1, in_channels.shape[0], 1, 1)
        index = index.expand(shape[0], -1, shape[2], shape[3])
        return torch.gather(t, 1, index)

    @staticmethod
    def scatter_channels(t, out_channels, channels):
        if torch.equal(out_channels, torch.linspace(0, channels - 1, channels, dtype=torch.int64)):
            return t
        shape = t.shape
        z = torch.zeros(1, dtype=t.dtype, device=t.device
                        ).view(1, 1, 1, 1).expand(shape[0], channels, shape[2], shape[3])
        index = out_channels.view(1, out_channels.shape[0], 1, 1).expand(shape[0], out_channels.shape[0],
                                                                         shape[2], shape[3])
        index = index.to(t.device)
        z = z.scatter(1, index, t)
        return z

    def gather_and_scatter(self, x, in_channels, out_channels):
        if 0 in x.size():
            return x
        gathered_x = self.gather_channels(x, in_channels)
        scattered_x = self.scatter_channels(gathered_x, out_channels, self.channels)
        return scattered_x

    def forward(self, input_a, input_b):
        self.input_a_shape = input_a.shape
        self.input_b_shape = input_b.shape
        input_a = self.gather_and_scatter(input_a, self.in_channels_a, self.out_channels_a)
        input_b = self.gather_and_scatter(input_b, self.in_channels_b, self.out_channels_b)
        result = add(input_a, input_b)
        self.output_image_shape = result.shape
        self.device = result.device
        return result

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


class FrozenAdder(nn.Module):
    def __init__(self, in_channels_a, out_channels_a, in_channels_b, out_channels_b, channels,
                 input_a_shape, input_b_shape, output_image_shape, device):
        super(FrozenAdder, self).__init__()

        self.skip_gather_a = torch.equal(in_channels_a,
                                         torch.linspace(0, input_a_shape[1] - 1, input_a_shape[1], dtype=torch.int64))
        if not self.skip_gather_a:
            gather_index_a = in_channels_a.view(1, in_channels_a.shape[0], 1, 1)
            self.gather_index_a = gather_index_a.expand(output_image_shape[0], -1,
                                                        output_image_shape[2], output_image_shape[3])

        self.skip_gather_b = torch.equal(in_channels_b,
                                         torch.linspace(0, input_b_shape[1] - 1, input_b_shape[1], dtype=torch.int64))
        if not self.skip_gather_b:
            gather_index_b = in_channels_b.view(1, in_channels_b.shape[0], 1, 1)
            self.gather_index_b = gather_index_b.expand(output_image_shape[0], -1,
                                                        output_image_shape[2], output_image_shape[3])

        self.skip_scatter_a = torch.equal(in_channels_a,
                                          torch.linspace(0, channels - 1, channels, dtype=torch.int64))
        if not self.skip_scatter_a:
            self.scatter_index_a = out_channels_a.view(
                1, out_channels_a.shape[0], 1, 1).expand(output_image_shape[0], out_channels_a.shape[0],
                                                         output_image_shape[2], output_image_shape[3])

        self.skip_scatter_b = torch.equal(in_channels_b,
                                          torch.linspace(0, channels - 1, channels, dtype=torch.int64))
        if not self.skip_scatter_b:
            self.scatter_index_b = out_channels_b.view(
                1, out_channels_b.shape[0], 1, 1).expand(output_image_shape[0], out_channels_b.shape[0],
                                                         output_image_shape[2], output_image_shape[3])

        self.zeros = torch.zeros(1, device=device).view(1, 1, 1, 1).expand(
            output_image_shape[0], channels, output_image_shape[2], output_image_shape[3])

    def forward(self, input_a, input_b):
        if 0 not in input_a.shape:
            if not self.skip_gather_a:
                input_a = torch.gather(input_a, 1, self.gather_index_a)
            if not self.skip_scatter_a:
                input_a = self.zeros.scatter(1, self.scatter_index_a, input_a)
        if 0 not in input_b.shape:
            if not self.skip_gather_b:
                input_b = torch.gather(input_b, 1, self.gather_index_b)
            if not self.skip_scatter_b:
                input_b = self.zeros.scatter(1, self.scatter_index_b, input_b)
        return add(input_a, input_b)


def freeze_adders(model, image_shape):
    def freeze_adders_(network):
        for n, m in network.named_children():
            if isinstance(m, Adder):
                if 0 in m.output_image_shape:
                    setattr(network, n, EmptyLayer().to(m.device))
                else:
                    setattr(network, n,
                            FrozenAdder(m.in_channels_a, m.out_channels_a,
                                        m.in_channels_b, m.out_channels_b,
                                        m.channels, m.input_a_shape, m.input_b_shape,
                                        m.output_image_shape, m.device))
            else:
                freeze_adders_(m)
    device = None
    for p in model.parameters():
        device = p.device
        break
    model(torch.zeros(image_shape).to(device))
    freeze_adders_(model)
