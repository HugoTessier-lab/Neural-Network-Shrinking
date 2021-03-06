import torch.nn as nn
import torch.nn.functional as F
from shrinking.custom_operators import Adder, add, freeze_adders


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, adder=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if adder:
            self.adder = Adder(planes)
        else:
            self.adder = add

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.adder(out, self.shortcut(x))
        out = F.relu(out)
        return out

    def get_safe_layers(self):
        return [self.conv1, self.bn1]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, adder=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if adder:
            self.adder = Adder(planes * self.expansion)
        else:
            self.adder = add

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.adder(out, self.shortcut(x))
        out = F.relu(out)
        return out

    def get_safe_layers(self):
        return [self.conv1, self.bn1, self.conv2, self.bn2]


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64, adder=False):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.is_imagenet = num_classes == 1000

        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.is_imagenet:
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        else:  # CIFAR
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1, adder=adder)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2, adder=adder)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2, adder=adder)
        self.layer4 = None
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2, adder=adder)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

        self.frozen = False
        self.pool_size = None

    def _make_layer(self, block, planes, num_blocks, stride, adder=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, adder))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.is_imagenet:
            out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.layer4:
            out = self.layer4(out)
        if 0 not in out.size():
            if not self.frozen:
                self.pool_size = out.size()
            out = F.avg_pool2d(out, self.pool_size[3])
            out = out.view(self.pool_size[0], -1)
        out = self.linear(out)
        return out

    def freeze(self, image_shape):
        freeze_adders(self, image_shape)
        self.frozen = True

    def get_safe_layers(self):
        layers = []
        for m in self.modules():
            if isinstance(m, BasicBlock) or isinstance(m, Bottleneck):
                layers.extend(m.get_safe_layers())
        return layers


def resnet18(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet34(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet50(num_classes=10, in_planes=64, adder=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet101(num_classes=10, in_planes=64, adder=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet152(num_classes=10, in_planes=64, adder=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet20(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet32(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet44(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet56(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, in_planes=in_planes, adder=adder)


def resnet110(num_classes=10, in_planes=64, adder=False):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, in_planes=in_planes, adder=adder)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def resnet_model(model, num_classes=10, in_planes=64, adder=False):
    if model == 'resnet18':
        return resnet18(num_classes, in_planes, adder)
    elif model == 'resnet34':
        return resnet34(num_classes, in_planes, adder)
    elif model == 'resnet50':
        return resnet50(num_classes, in_planes, adder)
    elif model == 'resnet101':
        return resnet101(num_classes, in_planes, adder)
    elif model == 'resnet152':
        return resnet152(num_classes, in_planes, adder)
    elif model == 'resnet20':
        return resnet20(num_classes, in_planes, adder)
    elif model == 'resnet32':
        return resnet32(num_classes, in_planes, adder)
    elif model == 'resnet44':
        return resnet44(num_classes, in_planes, adder)
    elif model == 'resnet56':
        return resnet56(num_classes, in_planes, adder)
    elif model == 'resnet110':
        return resnet110(num_classes, in_planes, adder)
    else:
        raise ValueError
