import torch
import torch.nn as nn
from utils.pruning.custom_layers import Gate, ChannelMapper, MutableChannelMapper, unfreeze_mapper, EmptyLayer
import copy
from utils.pruning.layer_shrinker import shrink_identity, shrink_gate, shrink_bn, shrink_conv


class Pruner:
    def create_mock(self, network):
        for n, m in network.named_children():
            if isinstance(m, ChannelMapper):
                setattr(network, n, unfreeze_mapper(m).to(self.device))
            elif len([i for i in m.named_children()]) == 0:
                pass
            else:
                self.create_mock(m)

    @staticmethod
    def remove_bias(network):
        for m in network.modules():
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias = None
                    setattr(m, 'removed_bias', True)

    def __init__(self, model, image_shape):
        self.mock = copy.deepcopy(model)
        self.remove_bias(self.mock)
        self.device = None
        for p in self.mock.parameters():
            if self.device is None:
                self.device = p.device
            p.data.abs_()
            p.data += (p.data == 0).float() * p.data.mean()
            if p.data.sum() == 0:
                p.data.fill_(1)
        for m in self.mock.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.fill_(0)
                m.running_var.fill_(1)
        self.create_mock(self.mock)

        def backward_hook(self, input, output):
            new = []
            for i in input:
                if i is not None:
                    new.append((i != 0).float())
                else:
                    new.append(None)
            return tuple(new)

        def forward_hook(self, input, output):
            if isinstance(output, torch.Tensor):
                return torch.tanh(output)
            else:
                return [torch.tanh(o) for o in output]

        for p in self.mock.modules():
            p.register_forward_hook(forward_hook)
            p.register_full_backward_hook(backward_hook)
        self.image_shape = image_shape

    def apply_mask(self, mask):
        i = 0
        for m in self.mock.modules():
            if isinstance(m, Gate):
                m.weight.data = mask[i]
                i += 1

    def _grad_to_weights(self):
        for m in self.mock.modules():
            if hasattr(m, 'weight'):
                m.weight.data = m.weight.data * ((m.weight.data != 0) & (m.weight.grad != 0)).float()

    def compute_pruning(self):
        previous_count = 0
        new_count = self.get_params_count()
        while new_count != previous_count:
            previous_count = new_count
            image = torch.ones(self.image_shape).to(self.device)
            self.mock.zero_grad()
            out = self.mock(image)
            back = out.mean()
            back.backward()
            new_count = self.get_params_count()
            self._grad_to_weights()

    def get_params_count(self):
        unpruned = 0
        for m in self.mock.modules():
            if not isinstance(m, MutableChannelMapper):
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        unpruned += ((m.weight.data.flatten() != 0) & (m.weight.grad.flatten() != 0)).sum()
                    else:
                        unpruned += (m.weight.data.flatten() != 0).sum()
                if hasattr(m, 'removed_bias'):
                    if m.weight.grad is not None:
                        weights = ((m.weight.data != 0) & (m.weight.grad != 0))
                    else:
                        weights = (m.weight.data != 0)
                    if len(weights.shape) > 1:
                        unpruned += (torch.mean(weights.float(),
                                                dim=[i for i in range(1, len(weights.shape))]) != 0).sum()
                    else:
                        unpruned += weights.sum()

        return unpruned

    def _shrink_model(self, model, mock):
        for (m, m_mock) in zip(model.children(), mock.children()):
            if isinstance(m, nn.Conv2d):
                shrink_conv(m, m_mock)
            elif isinstance(m, nn.BatchNorm2d):
                shrink_bn(m, m_mock)
            elif isinstance(m, Gate):
                shrink_gate(m, m_mock)
            elif isinstance(m, ChannelMapper):
                shrink_identity(m, m_mock)
            else:
                self._shrink_model(m, m_mock)

    def purge_empty_layers(self, model):
        for n, m in model.named_children():
            if isinstance(m, ChannelMapper):
                if 0 in m.in_channels.size() or 0 in m.out_channels.size():
                    setattr(model, n, EmptyLayer().to(self.device))
            if hasattr(m, 'weight'):
                if 0 in m.weight.data.size():
                    if hasattr(m, 'bias'):
                        if m.bias is not None:
                            if 0 in m.bias.data.size():
                                setattr(model, n, EmptyLayer().to(self.device))
                        else:
                            setattr(model, n, EmptyLayer().to(self.device))
                    else:
                        setattr(model, n, EmptyLayer().to(self.device))
            else:
                self.purge_empty_layers(m)

    def shrink_model(self, model):
        self._shrink_model(model, self.mock)
        self.purge_empty_layers(model)
