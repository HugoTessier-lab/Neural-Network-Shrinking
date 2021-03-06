import torch
import torch.nn as nn
from shrinking.custom_operators import Adder, MutableAdder, unfreeze_adder, EmptyLayer
import copy
from shrinking.layer_shrinker import shrink_identity, shrink_bn, shrink_conv, shrink_linear


def backward_hook(self, input, output):
    new = []
    for i in input:
        if i is not None:
            new.append((i != 0).type(i.dtype))
        else:
            new.append(None)
    return tuple(new)


def forward_hook(self, input, output):
    if isinstance(self, nn.Conv2d):
        if isinstance(output, torch.Tensor):
            return torch.log(output + 1)
        else:
            return [torch.log(o + 1) for o in output]


class Mock:
    def __init__(self, model, device, dtype, remove_biases):
        self.device = device
        self.mock = copy.deepcopy(model)
        self.mock = self.mock.type(dtype)
        self.mock.eval()
        self.remove_activations(self.mock)
        if remove_biases:
            self.remove_bias()
        self.neutralize_batchnorms()
        self.adjust_weights()
        self.unfreeze_adders(self.mock, dtype)
        self.insert_hooks()

    def remove_activations(self, network):
        for n, m in network.named_children():
            if isinstance(m, nn.ReLU):
                setattr(network, n, nn.Identity())
            else:
                self.remove_activations(m)

    def remove_bias(self):
        for m in self.mock.modules():
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias = None
                    setattr(m, 'removed_bias', True)

    def neutralize_batchnorms(self):
        for m in self.mock.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()
                m.eval()

    def unfreeze_adders(self, network, dtype):
        for n, m in network.named_children():
            if isinstance(m, Adder):
                setattr(network, n, unfreeze_adder(m).to(self.device).type(dtype))
            elif len([i for i in m.named_children()]) == 0:
                pass
            else:
                self.unfreeze_adders(m, dtype)

    def adjust_weights(self):
        for p in self.mock.parameters():
            p.data.fill_(1)

    def insert_hooks(self):
        for p in self.mock.modules():
            p.register_forward_hook(forward_hook)
            p.register_full_backward_hook(backward_hook)

    def generate_masked_mock(self, mask):
        mock = copy.deepcopy(self.mock)
        i = 0
        for m in mock.modules():
            if hasattr(m, 'weight') and not isinstance(m, MutableAdder) \
                    and not isinstance(m, MutableAdder.MappingConvolution):
                m.weight.data = (m.weight.data * mask[i]).type(m.weight.data.dtype)
                i += 1
            if hasattr(m, 'removed_bias'):
                i += 1
            else:
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        m.bias.data = (m.bias.data * mask[i]).type(m.bias.data.dtype)
                        i += 1
        return mock


class Pruner:
    def __init__(self, model, image_shape, device, dtype=torch.float32, remove_biases=True):
        self.mock = Mock(model, device, dtype, remove_biases=remove_biases)
        self.image_shape = image_shape
        self.device = device
        self.count = None
        self.masked_mock = None
        self.dtype = dtype

    @staticmethod
    def grad_to_weights(model):
        for m in model.modules():
            if hasattr(m, 'weight'):
                m.weight.data = (m.weight.data * ((m.weight.data != 0) & (m.weight.grad != 0))
                                 ).type(m.weight.data.dtype)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data = (m.bias.data * ((m.bias.data != 0) & (m.bias.grad != 0))).type(m.bias.data.dtype)

    def apply_mask(self, mask):
        self.masked_mock = self.mock.generate_masked_mock(mask)
        previous_count = 0
        new_count = self.compute_params_count(self.masked_mock)
        while new_count != previous_count:
            previous_count = new_count
            image = torch.ones(self.image_shape).to(self.device).type(self.dtype)
            self.masked_mock.zero_grad()
            out = self.masked_mock(image)
            back = out.mean()
            back.backward()
            new_count = self.compute_params_count(self.masked_mock)
            self.grad_to_weights(self.masked_mock)
        self.count = new_count

    @staticmethod
    def compute_params_count(model):
        unpruned = 0
        for m in model.modules():
            if not isinstance(m, MutableAdder) and not isinstance(m, MutableAdder.MappingConvolution):
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
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            unpruned += ((m.bias.data != 0) & (m.bias.grad != 0)).sum()
                        else:
                            unpruned += (m.bias.data != 0).sum()

        return unpruned

    def get_params_count(self):
        return self.count

    def get_mask(self):
        masks = []
        for m in self.masked_mock.modules():
            if not isinstance(m, MutableAdder) and not isinstance(m, MutableAdder.MappingConvolution):
                if hasattr(m, 'weight'):
                    if m.weight.grad is not None:
                        masks.append((m.weight.data != 0) & (m.weight.grad != 0))
                    else:
                        masks.append(m.weight.data != 0)
                if hasattr(m, 'removed_bias'):
                    if m.weight.grad is not None:
                        weights = ((m.weight.data != 0) & (m.weight.grad != 0))
                    else:
                        weights = (m.weight.data != 0)
                    if len(weights.shape) > 1:
                        masks.append(torch.mean(weights.float(), dim=[i for i in range(1, len(weights.shape))]) != 0)
                    else:
                        masks.append(weights)
                if hasattr(m, 'bias'):
                    if m.bias is not None:
                        if m.bias.grad is not None:
                            masks.append((m.bias.data != 0) & (m.bias.grad != 0))
                        else:
                            masks.append(m.bias.data != 0)
        return masks

    def _shrink_model(self, model, mock):
        for (m, m_mock) in zip(model.children(), mock.children()):
            if isinstance(m, nn.Conv2d):
                shrink_conv(m, m_mock)
            elif isinstance(m, nn.BatchNorm2d):
                shrink_bn(m, m_mock)
            elif isinstance(m, nn.Linear):
                shrink_linear(m, m_mock)
            elif isinstance(m, Adder):
                shrink_identity(m, m_mock)
            else:
                self._shrink_model(m, m_mock)

    def purge_empty_layers(self, model):
        for n, m in model.named_children():
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
        self._shrink_model(model, self.masked_mock)
        self.purge_empty_layers(model)
        for p in model.parameters():
            p.grad = torch.zeros(p.shape).to(p.device)
