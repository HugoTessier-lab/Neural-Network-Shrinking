import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False, bn_after=self.bn1)
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetModel(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNetModel, self).__init__()
        self.in_planes = in_planes
        self.is_imagenet = num_classes == 1000

        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.is_imagenet:
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        else:  # CIFAR
            self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        size = 224 if self.is_imagenet else 32
        self.post_conv1_dim = (size, size)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = None
        if len(num_blocks) == 4:
            self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(in_planes * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_planes=64):
        super(ResNet, self).__init__()
        self.model = ResNetModel(block, num_blocks, num_classes, in_planes)
        self.module = self.model
        self.num_block = num_blocks
        self.num_classes = num_classes

    def forward(self, x):
        return self.model(x)


def resnet18(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_planes=in_planes)


def resnet34(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes)


def resnet50(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_planes=in_planes)


def resnet101(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_planes=in_planes)


def resnet152(num_classes=10, in_planes=64):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_planes=in_planes)


def resnet20(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, in_planes=in_planes)


def resnet32(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, in_planes=in_planes)


def resnet44(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, in_planes=in_planes)


def resnet56(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, in_planes=in_planes)


def resnet110(num_classes=10, in_planes=64):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, in_planes=in_planes)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110']


def resnet_model(model, num_classes=10, in_planes=64):
    if model == 'resnet18':
        return resnet18(num_classes, in_planes)
    elif model == 'resnet34':
        return resnet34(num_classes, in_planes)
    elif model == 'resnet50':
        return resnet50(num_classes, in_planes)
    elif model == 'resnet101':
        return resnet101(num_classes, in_planes)
    elif model == 'resnet152':
        return resnet152(num_classes, in_planes)
    elif model == 'resnet20':
        return resnet20(num_classes, in_planes)
    elif model == 'resnet32':
        return resnet32(num_classes, in_planes)
    elif model == 'resnet44':
        return resnet44(num_classes, in_planes)
    elif model == 'resnet56':
        return resnet56(num_classes, in_planes)
    elif model == 'resnet110':
        return resnet110(num_classes, in_planes)
    else:
        raise ValueError
