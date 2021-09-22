import torch.nn as nn
from models.shufflenet import ShuffleNetV2
from models.mobilenetv2 import MobileNetV2
from models.resnet import ResNet
import torch


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False, concat=False):
        super(UnetBlock, self).__init__()

        self.concat = concat

        self.transpose = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, bias=False)
        self.conv1 = nn.Conv2d(out_channels if not concat else 2 * out_channels, out_channels, 3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, side_input):
        x = self.transpose(x)
        x = x[:, :, 1:-1, 1:-1]
        if self.batchnorm:
            x = self.bn1(x)
        if self.concat:
            x = torch.cat((x, side_input), dim=1)
        else:
            x = x + side_input
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn2(x)
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn3(x)
        return x


class UnetBlockVariant(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=False, concat=False):
        super(UnetBlockVariant, self).__init__()

        self.concat = concat

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.transpose = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, bias=False)
        self.conv2 = nn.Conv2d(out_channels if not concat else 2 * out_channels, out_channels, 1, bias=False)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, side_input):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.transpose(x)
        x = x[:, :, 1:-1, 1:-1]
        if self.batchnorm:
            x = self.bn2(x)
        if self.concat:
            x = torch.cat((x, side_input), dim=1)
        else:
            x = x + side_input
        x = self.conv2(x)
        if self.batchnorm:
            x = self.bn3(x)
        return x


class Unet(nn.Module):
    def __init__(self, encoder, num_classes=19, variant=False, batchnorm=False, concat=False):
        super(Unet, self).__init__()

        self.encoder = encoder
        encoder_dims = self.encoder.get_dimensions()
        encoder_dims.reverse()
        if isinstance(self.encoder, MobileNetV2) or isinstance(self.encoder, ShuffleNetV2):
            encoder_dims.pop(1)
        if isinstance(self.encoder, ResNet):
            encoder_dims.pop(-2)

        if not variant:
            block = UnetBlock
        else:
            block = UnetBlockVariant

        self.blocks = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            b = block(encoder_dims[i], encoder_dims[i + 1], batchnorm=batchnorm, concat=concat)
            self.blocks.append(b)

        if isinstance(self.encoder, ShuffleNetV2) or isinstance(self.encoder, ResNet):
            self.transpose = nn.ConvTranspose2d(encoder_dims[-1], encoder_dims[-1], 8, stride=4, bias=False)
        else:
            self.transpose = nn.ConvTranspose2d(encoder_dims[-1], encoder_dims[-1], 4, stride=2, bias=False)

        self.conv = nn.Conv2d(encoder_dims[-1], num_classes, 1, bias=False)

        self.batchnorm = batchnorm
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(encoder_dims[-1])

    def forward(self, x):
        outputs = self.encoder(x)
        outputs.reverse()
        if isinstance(self.encoder, MobileNetV2) or isinstance(self.encoder, ShuffleNetV2):
            outputs.pop(1)
        if isinstance(self.encoder, ResNet):
            outputs.pop(-2)

        x = outputs[0]
        for i in range(len(outputs) - 1):
            x = self.blocks[i](x, outputs[i + 1])

        x = self.transpose(x)
        if isinstance(self.encoder, ShuffleNetV2) or isinstance(self.encoder, ResNet):
            x = x[:, :, 2:-2, 2:-2]
        else:
            x = x[:, :, 1:-1, 1:-1]

        if self.batchnorm:
            x = self.bn1(x)
        x = self.conv(x)

        return x
