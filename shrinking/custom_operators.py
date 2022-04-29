import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.onnx.symbolic_helper import parse_args


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor([]), requires_grad=False)

    def forward(self, x, *args):
        return torch.Tensor([]).to(x.device)


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
        if torch.equal(in_channels, torch.linspace(0, t.shape[1] - 1, t.shape[1],
                                                   dtype=torch.int64).to(in_channels.device)):
            return t
        shape = t.shape
        index = in_channels.view(1, in_channels.shape[0], 1, 1)
        index = index.expand(shape[0], -1, shape[2], shape[3])
        return torch.gather(t, 1, index)

    @staticmethod
    def scatter_channels(t, out_channels, channels):
        if torch.equal(out_channels, torch.linspace(0, channels - 1, channels,
                                                    dtype=torch.int64).to(out_channels.device)):
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


@parse_args('v', 'v', 'v')
def scatternd(g, self, index, src):
    return g.op("ScatterND", self, index, src)


class MyScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, indices, zeros):
        indices = indices.view(indices.size(0), 1, 1, 1).expand(src.shape)
        return zeros.scatter(0, indices, src)

    @staticmethod
    def symbolic(g, src, indices, zeros):
        return scatternd(g, zeros, indices, src)


class FrozenAdder(torch.nn.Module):
    def __init__(self, in_channels_a, out_channels_a, in_channels_b, out_channels_b, channels, device):
        super(FrozenAdder, self).__init__()

        if not torch.equal(in_channels_a, torch.linspace(0, channels - 1, channels, dtype=torch.int64).to(device)):
            self.scatter_a = out_channels_a[:, None].to(device)
        else:
            self.scatter_a = None

        if not torch.equal(in_channels_b, torch.linspace(0, channels - 1, channels, dtype=torch.int64).to(device)):
            self.scatter_b = out_channels_b[:, None].to(device)
        else:
            self.scatter_b = None

        self.channels = channels

    def forward(self, a, b):
        if 0 not in a.shape:
            shape = a.shape
        elif 0 not in b.shape:
            shape = b.shape
        else:
            return torch.Tensor([]).to(a.device)
        zeros = torch.Tensor([0]).view(1, 1, 1, 1).expand(self.channels, shape[0], shape[2], shape[3]).to(a.device)
        if 0 not in a.shape:
            if self.scatter_a is not None:
                out1 = MyScatterFunction.apply(a.transpose(0, 1), self.scatter_a, zeros).transpose(0, 1)
            else:
                out1 = a
        else:
            out1 = None
        if 0 not in b.shape:
            if self.scatter_b is not None:
                out2 = MyScatterFunction.apply(b.transpose(0, 1), self.scatter_b, zeros).transpose(0, 1)
            else:
                out2 = b
        else:
            out2 = None
        if out1 is None and out2 is not None:
            return out2
        elif out1 is not None and out2 is None:
            return out1
        elif out1 is None and out2 is None:
            return torch.Tensor([]).to(a.device)
        else:
            return out1 + out2


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
                                        m.channels, m.device))
            else:
                freeze_adders_(m)

    device = None
    for p in model.parameters():
        device = p.device
        break
    model(torch.zeros(image_shape).to(device))
    freeze_adders_(model)
