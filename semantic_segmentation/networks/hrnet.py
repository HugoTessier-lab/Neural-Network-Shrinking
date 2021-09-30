import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url

__all__ = ['hrnet18', 'hrnet32', 'hrnet48']

from pruning.pruner.custom_layers import Gate, create_channel_mapper

model_urls = {
    'hrnet18_imagenet': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5w'
                        'DgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJ'
                        'bbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32_imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx'
                        '0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQr'
                        'TaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48_imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6L'
                        'K_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_S'
                        'fNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ'
                          '6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ'
                          '-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
    'hrnet48_ocr_cityscapes': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk'
                              '8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWo'
                              't0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}


def add(a, b):
    if 0 not in a.size() and 0 in b.size():
        return a
    elif 0 in a.size() and 0 not in b.size():
        return b
    elif 0 not in a.size() and 0 not in b.size():
        return a + b
    else:
        return torch.Tensor([]).to(a.device)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, gates=False, mappers=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            if mappers:
                self.downsample = create_channel_mapper(inplanes)
            else:
                self.downsample = None
        self.stride = stride

        if gates:
            self.gate1 = Gate(planes)
            self.gate2 = Gate(planes)
        else:
            self.gate1 = None
            self.gate2 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate1 is not None:
            out = self.gate1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = add(out, identity)
        out = self.relu(out)
        if self.gate2 is not None:
            out = self.gate2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, gates=False, mappers=False):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride=stride),
                nn.BatchNorm2d(planes * self.expansion),
            )
        else:
            if mappers:
                self.downsample = create_channel_mapper(inplanes)
            else:
                self.downsample = None
        self.stride = stride

        if gates:
            self.gate1 = Gate(width)
            self.gate2 = Gate(planes)
            self.gate3 = Gate(planes * self.expansion)
        else:
            self.gate1 = None
            self.gate2 = None
            self.gate3 = None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.gate1 is not None:
            out = self.gate1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.gate2 is not None:
            out = self.gate2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = add(out, identity)
        out = self.relu(out)
        if self.gate3 is not None:
            out = self.gate3(out)

        return out


class FuseBlock(nn.Module):
    def __init__(self, i, num_branches, num_inchannels, gates=False, mappers=False):
        super(FuseBlock, self).__init__()
        self.num_branches = num_branches
        self.i = i
        fuse_layer = []
        for j in range(num_branches):
            if j > i:
                fuse_layer.append(nn.Sequential(
                    conv1x1(num_inchannels[j], num_inchannels[i]),
                    nn.BatchNorm2d(num_inchannels[i])))
            elif j == i:
                fuse_layer.append(None)
            else:
                conv3x3s = []
                for k in range(i - j):
                    if k == i - j - 1:
                        num_outchannels_conv3x3 = num_inchannels[i]
                        conv3x3s.append(nn.Sequential(
                            conv3x3(num_inchannels[j], num_outchannels_conv3x3, stride=2),
                            nn.BatchNorm2d(num_outchannels_conv3x3)))
                    else:
                        num_outchannels_conv3x3 = num_inchannels[j]
                        modules = [conv3x3(num_inchannels[j], num_outchannels_conv3x3, stride=2),
                                   nn.BatchNorm2d(num_outchannels_conv3x3),
                                   nn.ReLU()]
                        if gates:
                            modules.append(Gate(num_outchannels_conv3x3))
                        conv3x3s.append(nn.Sequential(*modules))
                fuse_layer.append(nn.Sequential(*conv3x3s))
        self.fuse_layer = nn.ModuleList(fuse_layer)

        if gates:
            self.gate = Gate(num_inchannels[i])
        else:
            self.gate = None
        if mappers:
            self.identity = create_channel_mapper(num_inchannels[i])
        else:
            self.identity = None
        self.resolution = None

    def forward(self, x):
        y = (self.identity(x[0]) if self.identity is not None else x[0]) if self.i == 0 else self.fuse_layer[0](x[0])
        for j in range(1, self.num_branches):
            if self.i == j:
                y = add(y, self.identity(x[j]) if self.identity is not None else x[j])
            elif j > self.i:
                if 0 in x[self.i].size():
                    if self.resolution is None:
                        print('ERROR : missing resolution information for interpolation')
                        raise RuntimeError
                    width_output = self.resolution[0]
                    height_output = self.resolution[1]
                else:
                    width_output = x[self.i].shape[-1]
                    height_output = x[self.i].shape[-2]
                    self.resolution = [width_output, height_output]
                result = self.fuse_layer[j](x[j])
                if 0 not in result.size():
                    interpolation = F.interpolate(result,
                                                  size=[height_output, width_output],
                                                  mode='bilinear', align_corners=False)
                    y = add(y, interpolation)
            else:
                y = add(y, self.fuse_layer[j](x[j]))
        y = F.relu(y)
        if self.gate is not None:
            y = self.gate(y)
        return y


class FuseStage(nn.Module):
    def __init__(self, num_branches, num_inchannels, multi_scale_output, gates=False, mappers=False):
        super(FuseStage, self).__init__()

        self.num_branches = num_branches
        if num_branches == 1:
            self.layer = None

        fuse_layers = []
        for i in range(num_branches if multi_scale_output else 1):
            fuse_layers.append(FuseBlock(i, num_branches, num_inchannels, gates=gates, mappers=mappers))

        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            x_fuse.append(self.fuse_layers[i](x))
        return x_fuse


class HighResolutionBranches(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_channels, num_inchannels, gates=False, mappers=False):
        super(HighResolutionBranches, self).__init__()
        self.num_inchannels = num_inchannels
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels, gates=gates, mappers=mappers))
        self.branches = nn.ModuleList(branches)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1, gates=False, mappers=False):
        layers = list()
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride,
                            gates=gates, mappers=mappers))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index],
                                gates=gates, mappers=mappers))

        return nn.Sequential(*layers)

    def __getitem__(self, item):
        return self.branches[item]

    def get_num_inchannels(self):
        return self.num_inchannels


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, gates=False, mappers=False):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = HighResolutionBranches(num_branches, blocks, num_blocks, num_channels, self.num_inchannels,
                                               gates=gates, mappers=mappers)
        self.num_inchannels = self.branches.get_num_inchannels()
        self.fuse_layers = FuseStage(num_branches, num_inchannels, multi_scale_output, gates=gates, mappers=mappers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = self.fuse_layers(x)
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class TransitionLayer(nn.Module):
    def __init__(self, num_channels_pre_layer, num_channels_cur_layer, previous_num_branches, cfg, gates=False):
        super(TransitionLayer, self).__init__()
        self.previous_num_branches = previous_num_branches
        self.cfg = cfg

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        self.num_channels_cur_layer = num_channels_cur_layer
        self.num_channels_pre_layer = num_channels_pre_layer
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    modules = [conv3x3(num_channels_pre_layer[i], num_channels_cur_layer[i]),
                               nn.BatchNorm2d(num_channels_cur_layer[i]),
                               nn.ReLU()]
                    if gates:
                        modules.append(Gate(num_channels_cur_layer[i]))
                    transition_layers.append(nn.Sequential(*modules))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    modules = [conv3x3(inchannels, outchannels, stride=2),
                               nn.BatchNorm2d(outchannels),
                               nn.ReLU()]
                    if gates:
                        modules.append(Gate(outchannels))
                    conv3x3s.append(nn.Sequential(*modules))

                transition_layers.append(nn.Sequential(*conv3x3s))

        self.transition_layers = nn.ModuleList(transition_layers)

    def forward(self, input_list):
        output_list = []
        for i in range(self.cfg['NUM_BRANCHES']):
            if self.transition_layers[i] is not None:
                if i < self.previous_num_branches:
                    output_list.append(self.transition_layers[i](input_list[i]))
                else:
                    output_list.append(self.transition_layers[i](input_list[-1]))
            else:
                output_list.append(input_list[i])
        return output_list


class HighResolutionStage(nn.Module):
    def __init__(self, cfg, pre_stage_channels, previous_num_branches, gates=False, mappers=False):
        super(HighResolutionStage, self).__init__()
        self.previous_num_branches = previous_num_branches
        self.cfg = cfg
        num_channels = self.cfg['NUM_CHANNELS']
        block = blocks_dict[self.cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition = TransitionLayer(pre_stage_channels, num_channels, previous_num_branches, cfg, gates=gates)
        self.stage, self.pre_stage_channels = self._make_stage(self.cfg, num_channels, gates=gates, mappers=mappers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, gates=False, mappers=False):
        num_modules = layer_config['NUM_MODULES']
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(num_branches=layer_config['NUM_BRANCHES'],
                                     blocks=blocks_dict[layer_config['BLOCK']],
                                     num_blocks=layer_config['NUM_BLOCKS'],
                                     num_inchannels=num_inchannels,
                                     num_channels=layer_config['NUM_CHANNELS'],
                                     fuse_method=layer_config['FUSE_METHOD'],
                                     multi_scale_output=not (not multi_scale_output and i == num_modules - 1),
                                     gates=gates,
                                     mappers=mappers))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def get_pre_stage_channels(self):
        return self.pre_stage_channels

    def forward(self, input_list):
        output_list = self.transition(input_list)
        output_list = self.stage(output_list)
        return output_list


class ResNetStage(nn.Module):
    def __init__(self, cfg, inplanes, gates=False, mappers=False):
        super(ResNetStage, self).__init__()
        self.cfg = cfg
        num_channels = self.cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.cfg['BLOCK']]
        num_blocks = self.cfg['NUM_BLOCKS'][0]
        self.pre_stage_channels = [block.expansion * num_channels]
        self.layer = self._make_layer(block, inplanes, num_channels, num_blocks, gates=gates, mappers=mappers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, gates=False, mappers=False):
        layers = list()
        layers.append(block(inplanes, planes, stride, gates=gates, mappers=mappers))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, gates=gates, mappers=mappers))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    def get_pre_stage_channels(self):
        return self.pre_stage_channels


class HighResolutionNet(nn.Module):
    def __init__(self, cfg, num_classes=19, gates=False, mappers=False):
        super(HighResolutionNet, self).__init__()

        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        if gates:
            self.gate1 = Gate(64)
        else:
            self.gate1 = None
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        if gates:
            self.gate2 = Gate(64)
        else:
            self.gate2 = None
        self.relu = nn.ReLU()

        self.layer1 = ResNetStage(cfg['STAGE1'], inplanes=64, gates=gates, mappers=mappers)
        self.stage2 = HighResolutionStage(cfg['STAGE2'], self.layer1.get_pre_stage_channels(),
                                          cfg['STAGE1']['NUM_BRANCHES'], gates=gates, mappers=mappers)
        self.stage3 = HighResolutionStage(cfg['STAGE3'], self.stage2.get_pre_stage_channels(),
                                          cfg['STAGE2']['NUM_BRANCHES'], gates=gates, mappers=mappers)
        self.stage4 = HighResolutionStage(cfg['STAGE4'], self.stage3.get_pre_stage_channels(),
                                          cfg['STAGE3']['NUM_BRANCHES'], gates=gates, mappers=mappers)

        last_inp_channels = int(np.sum(self.stage4.get_pre_stage_channels()))
        mod = [conv1x1(in_planes=last_inp_channels, out_planes=last_inp_channels, stride=1, bias=True),
               nn.BatchNorm2d(last_inp_channels),
               nn.ReLU()]
        if gates:
            mod.append(Gate(last_inp_channels))
        mod.append(conv1x1(in_planes=last_inp_channels, out_planes=num_classes, stride=1, bias=True))
        self.last_layer = nn.Sequential(*mod)
        self.num_classes = num_classes
        self.resolution = None

    def forward(self, x):
        original_shape = x.shape[-2:]
        original_device = x.device
        x = self.relu(self.bn1(self.conv1(x)))
        if self.gate1 is not None:
            x = self.gate1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        if self.gate2 is not None:
            x = self.gate2(x)
        x = self.layer1(x)
        x = [x]

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        # Upsampling
        if 0 in x[0].size():
            if self.resolution is None:
                print('ERROR : missing resolution information for interpolation')
                raise RuntimeError
            x0_h, x0_w = self.resolution[0], self.resolution[1]
        else:
            x0_h, x0_w = x[0].size(2), x[0].size(3)
            self.resolution = [x0_h, x0_w]
        if 0 not in x1.size():
            x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        else:
            x1 = torch.Tensor([]).to(original_device)
        if 0 not in x2.size():
            x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        else:
            x2 = torch.Tensor([]).to(original_device)
        if 0 not in x3.size():
            x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        else:
            x3 = torch.Tensor([]).to(original_device)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        if 0 not in x.size():
            x = F.interpolate(x, size=original_shape, mode='bilinear', align_corners=False)
        return x


def _hrnet(arch, pretrained, progress, gates=False, mappers=False, **kwargs):
    model = HighResolutionNet(arch, gates=gates, mappers=mappers, **kwargs)
    if pretrained:
        model_url = arch['url']
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model


def hrnet18(pretrained=True, progress=True, gates=False, mappers=False, **kwargs):
    arch = {'url': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6Xu'
                   'QmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJH'
                   'gHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
            'FINAL_CONV_KERNEL': 1,
            'STAGE1': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 1,
                       'NUM_BLOCKS': [4],
                       'NUM_CHANNELS': [64],
                       'BLOCK': 'BOTTLENECK',
                       'FUSE_METHOD': 'SUM'},
            'STAGE2': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 2,
                       'NUM_BLOCKS': [4, 4],
                       'NUM_CHANNELS': [18, 36],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [4, 4, 4],
                       'NUM_CHANNELS': [18, 36, 72],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [4, 4, 4, 4],
                       'NUM_CHANNELS': [18, 36, 72, 144],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
            }
    return _hrnet(arch, pretrained, progress, gates=gates, mappers=mappers, **kwargs)


def hrnet32(pretrained=True, progress=True, gates=False, mappers=False, **kwargs):
    arch = {'url': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0W'
                   'Th42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjt'
                   'jMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
            'FINAL_CONV_KERNEL': 1,
            'STAGE1': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 1,
                       'NUM_BLOCKS': [4],
                       'NUM_CHANNELS': [64],
                       'BLOCK': 'BOTTLENECK',
                       'FUSE_METHOD': 'SUM'},
            'STAGE2': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 2,
                       'NUM_BLOCKS': [4, 4],
                       'NUM_CHANNELS': [32, 64],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [4, 4, 4],
                       'NUM_CHANNELS': [32, 64, 128],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [4, 4, 4, 4],
                       'NUM_CHANNELS': [32, 64, 128, 256],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
            }

    return _hrnet(arch, pretrained, progress, gates=gates, mappers=mappers, **kwargs)


def hrnet48(pretrained=True, progress=True, gates=False, mappers=False, **kwargs):
    arch = {'url': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHH'
                   'IYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF'
                   '5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ',
            'FINAL_CONV_KERNEL': 1,
            'STAGE1': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 1,
                       'NUM_BLOCKS': [4],
                       'NUM_CHANNELS': [64],
                       'BLOCK': 'BOTTLENECK',
                       'FUSE_METHOD': 'SUM'},
            'STAGE2': {'NUM_MODULES': 1,
                       'NUM_BRANCHES': 2,
                       'NUM_BLOCKS': [4, 4],
                       'NUM_CHANNELS': [48, 96],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE3': {'NUM_MODULES': 4,
                       'NUM_BRANCHES': 3,
                       'NUM_BLOCKS': [4, 4, 4],
                       'NUM_CHANNELS': [48, 96, 192],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'},
            'STAGE4': {'NUM_MODULES': 3,
                       'NUM_BRANCHES': 4,
                       'NUM_BLOCKS': [4, 4, 4, 4],
                       'NUM_CHANNELS': [48, 96, 192, 384],
                       'BLOCK': 'BASIC',
                       'FUSE_METHOD': 'SUM'}
            }
    return _hrnet(arch, pretrained, progress, gates=gates, mappers=mappers, **kwargs)
