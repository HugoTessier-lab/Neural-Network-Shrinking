import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url
import math

__all__ = ['hrnet18', 'hrnet32', 'hrnet48']

from shrinking.custom_operators import Adder, add, freeze_adders

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


def upsample(x, scale_factor):
    if 0 in x.size():
        return torch.tensor([]).to(x.device)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode='nearest')


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, adder=False):
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
            self.downsample = None
        self.stride = stride

        if adder:
            self.adder = Adder(planes)
        else:
            self.adder = add

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.adder(out, identity)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1, base_width=64, dilation=1, adder=False):
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
            self.downsample = None
        self.stride = stride

        if adder:
            self.adder = Adder(planes * self.expansion)
        else:
            self.adder = add

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.adder(out, identity)
        out = self.relu(out)

        return out


class FuseBlock(nn.Module):
    def __init__(self, i, num_branches, num_inchannels, adder=False):
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
                        conv3x3s.append(nn.Sequential(conv3x3(num_inchannels[j], num_outchannels_conv3x3, stride=2),
                                                      nn.BatchNorm2d(num_outchannels_conv3x3),
                                                      nn.ReLU()))
                fuse_layer.append(nn.Sequential(*conv3x3s))
        self.fuse_layer = nn.ModuleList(fuse_layer)
        self.adder = []
        for j in range(1, self.num_branches):
            if adder:
                a = Adder(num_inchannels[i])
            else:
                a = add
            self.adder.append(a)
        if adder:
            self.adder = torch.nn.ModuleList(self.adder)

    def forward(self, x):
        y = x[0] if self.i == 0 else self.fuse_layer[0](x[0])
        for j in range(1, self.num_branches):
            if self.i == j:
                y = self.adder[j - 1](y, x[j])
            elif j > self.i:
                result = self.fuse_layer[j](x[j])
                result = upsample(result, scale_factor=math.pow(2, j - self.i))
                y = self.adder[j - 1](result, y)
            else:
                y = self.adder[j - 1](y, self.fuse_layer[j](x[j]))
        y = F.relu(y)
        return y


class FuseStage(nn.Module):
    def __init__(self, num_branches, num_inchannels, multi_scale_output, adder=False):
        super(FuseStage, self).__init__()

        self.num_branches = num_branches
        if num_branches == 1:
            self.layer = None

        fuse_layers = []
        for i in range(num_branches if multi_scale_output else 1):
            fuse_layers.append(FuseBlock(i, num_branches, num_inchannels, adder=adder))

        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            x_fuse.append(self.fuse_layers[i](x))
        return x_fuse


class HighResolutionBranches(nn.Module):
    def __init__(self, num_branches, block, num_blocks, num_channels, num_inchannels, adder=False):
        super(HighResolutionBranches, self).__init__()
        self.num_inchannels = num_inchannels
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels, adder=adder))
        self.branches = nn.ModuleList(branches)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1, adder=False):
        layers = list()
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, adder=adder))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], adder=adder))

        return nn.Sequential(*layers)

    def __getitem__(self, item):
        return self.branches[item]

    def get_num_inchannels(self):
        return self.num_inchannels


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, adder=False):
        super(HighResolutionModule, self).__init__()

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = HighResolutionBranches(num_branches, blocks, num_blocks, num_channels, self.num_inchannels,
                                               adder=adder)
        self.num_inchannels = self.branches.get_num_inchannels()
        self.fuse_layers = FuseStage(num_branches, num_inchannels, multi_scale_output, adder=adder)

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
    def __init__(self, num_channels_pre_layer, num_channels_cur_layer, previous_num_branches, cfg):
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
                    transition_layers.append(
                        nn.Sequential(conv3x3(num_channels_pre_layer[i], num_channels_cur_layer[i]),
                                      nn.BatchNorm2d(num_channels_cur_layer[i]),
                                      nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(conv3x3(inchannels, outchannels, stride=2),
                                                  nn.BatchNorm2d(outchannels),
                                                  nn.ReLU()))

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
    def __init__(self, cfg, pre_stage_channels, previous_num_branches, adder=False):
        super(HighResolutionStage, self).__init__()
        self.previous_num_branches = previous_num_branches
        self.cfg = cfg
        num_channels = self.cfg['NUM_CHANNELS']
        block = blocks_dict[self.cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]

        self.transition = TransitionLayer(pre_stage_channels, num_channels, previous_num_branches, cfg)
        self.stage, self.pre_stage_channels = self._make_stage(self.cfg, num_channels, adder=adder)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True, adder=False):
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
                                     adder=adder))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def get_pre_stage_channels(self):
        return self.pre_stage_channels

    def forward(self, input_list):
        output_list = self.transition(input_list)
        output_list = self.stage(output_list)
        return output_list


class ResNetStage(nn.Module):
    def __init__(self, cfg, inplanes, adder=False):
        super(ResNetStage, self).__init__()
        self.cfg = cfg
        num_channels = self.cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.cfg['BLOCK']]
        num_blocks = self.cfg['NUM_BLOCKS'][0]
        self.pre_stage_channels = [block.expansion * num_channels]
        self.layer = self._make_layer(block, inplanes, num_channels, num_blocks, adder=adder)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, adder=False):
        layers = list()
        layers.append(block(inplanes, planes, stride, adder=adder))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, adder=adder))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

    def get_pre_stage_channels(self):
        return self.pre_stage_channels


class HighResolutionNet(nn.Module):
    def __init__(self, cfg, num_classes=19, adder=False):
        super(HighResolutionNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = ResNetStage(cfg['STAGE1'], inplanes=64, adder=adder)
        self.stage2 = HighResolutionStage(cfg['STAGE2'], self.layer1.get_pre_stage_channels(),
                                          cfg['STAGE1']['NUM_BRANCHES'], adder=adder)
        self.stage3 = HighResolutionStage(cfg['STAGE3'], self.stage2.get_pre_stage_channels(),
                                          cfg['STAGE2']['NUM_BRANCHES'], adder=adder)
        self.stage4 = HighResolutionStage(cfg['STAGE4'], self.stage3.get_pre_stage_channels(),
                                          cfg['STAGE3']['NUM_BRANCHES'], adder=adder)

        last_inp_channels = int(np.sum(self.stage4.get_pre_stage_channels()))
        self.last_layer = nn.Sequential(
            conv1x1(in_planes=last_inp_channels, out_planes=last_inp_channels, stride=1, bias=True),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(),
            conv1x1(in_planes=last_inp_channels, out_planes=num_classes, stride=1, bias=True))
        self.num_classes = num_classes

    def forward(self, x):
        if not (((x.size(2) & (x.size(2) - 1) == 0) and x.size(2) != 0)
                and ((x.size(3) & (x.size(3) - 1) == 0) and x.size(3) != 0)):
            print('Error: image resolution is not a power of two!')
            raise ValueError
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        x = [x]

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x1 = x[1]
        x2 = x[2]
        x3 = x[3]

        x1 = upsample(x1, scale_factor=2)
        x2 = upsample(x2, scale_factor=4)
        x3 = upsample(x3, scale_factor=8)

        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)
        x = upsample(x, scale_factor=4)
        return x

    def freeze(self, image_shape, value=True):
        freeze_adders(self, image_shape)


def _hrnet(arch, pretrained, progress, adder=False, **kwargs):
    model = HighResolutionNet(arch, adder=adder, **kwargs)
    if pretrained:
        model_url = arch['url']
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model_state_dict = model.state_dict()
        # model.load_state_dict(state_dict, strict=False)
        new_state_dict = {}
        for (k1, v1), (k2, v2) in zip(state_dict.items(), model_state_dict.items()):
            if 'incre' not in k1:
                new_state_dict[k2] = v1
            else:
                new_state_dict[k2] = v2

        model.load_state_dict(new_state_dict)
    return model


def hrnet18(pretrained=True, progress=True, adder=False, **kwargs):
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
    return _hrnet(arch, pretrained, progress, adder=adder, **kwargs)


def hrnet32(pretrained=True, progress=True, adder=False, **kwargs):
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

    return _hrnet(arch, pretrained, progress, adder=adder, **kwargs)


def hrnet48(pretrained=True, progress=True, adder=False, **kwargs):
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
    return _hrnet(arch, pretrained, progress, adder=adder, **kwargs)
