import torch
import torch.nn as nn
from torch.nn.functional import relu, adaptive_avg_pool2d, interpolate


class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=1, bias=False, padding=1, depthwise=False):
        super(Conv, self).__init__()
        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size, dilation=dilation, bias=bias,
                          padding=padding, groups=inplanes),
                nn.Conv2d(inplanes, planes, 1, bias=bias)
            )
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, dilation=dilation, bias=bias,
                                  padding=padding)

    def forward(self, x):
        return self.conv(x)


class Upsampler(nn.Module):
    def __init__(self, rate=None, convtranspose=False, planes=256):
        super(Upsampler, self).__init__()
        self.convtranspose = convtranspose
        self.rate = int(rate)
        if convtranspose:
            self.upsample = nn.ConvTranspose2d(planes, planes, self.rate * 2, stride=self.rate, bias=False)

    def forward(self, x, target_size=None):
        if self.convtranspose:
            output = self.upsample(x)
            if x.shape[-2:] == (1, 1):
                return output
            return output[:, :, self.rate // 2:-self.rate // 2, self.rate // 2:-self.rate // 2]
        else:
            return interpolate(x, size=target_size, mode='bilinear', align_corners=False)


class PyramidModule(nn.Module):
    def __init__(self, inplanes, planes=256, depthwise=False, output_stride=16):
        super(PyramidModule, self).__init__()
        if output_stride == 8:
            mult = 2
        elif output_stride == 16:
            mult = 1
        else:
            raise ValueError

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = Conv(inplanes, planes, kernel_size=3, dilation=6 * mult,
                          bias=False, padding=6 * mult, depthwise=depthwise)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = Conv(inplanes, planes, kernel_size=3, dilation=12 * mult,
                          bias=False, padding=12 * mult, depthwise=depthwise)
        self.bn3 = nn.BatchNorm2d(planes)

        self.conv4 = Conv(inplanes, planes, kernel_size=3, dilation=18 * mult,
                          bias=False, padding=18 * mult, depthwise=depthwise)
        self.bn4 = nn.BatchNorm2d(planes)

        self.conv5 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x1 = relu(self.bn1(self.conv1(x)))
        x2 = relu(self.bn2(self.conv2(x)))
        x3 = relu(self.bn3(self.conv3(x)))
        x4 = relu(self.bn4(self.conv4(x)))
        x5 = self.conv5(adaptive_avg_pool2d(x, (1, 1)))
        x5 = relu(self.bn5(x5))
        x5 = interpolate(x5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return x1, x2, x3, x4, x5


class FeaturesFuser(nn.Module):
    def __init__(self, planes=256, depthwise=False, hierarchical=False, inplanes=None):
        super(FeaturesFuser, self).__init__()
        self.hierarchical = hierarchical
        if hierarchical:
            self.conv2 = Conv(planes, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
            self.bn2 = nn.BatchNorm2d(planes)

            self.conv3 = Conv(planes, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
            self.bn3 = nn.BatchNorm2d(planes)

            self.conv4 = Conv(planes, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
            self.bn4 = nn.BatchNorm2d(planes)

            self.conv5 = Conv(planes, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
            self.bn5 = nn.BatchNorm2d(planes)

            self.conv_residual = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bn_residual = nn.BatchNorm2d(planes)

        self.conv6 = nn.Conv2d(planes * 5, planes, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm2d(planes)

    def forward(self, x1, x2, x3, x4, x5, x0=None):
        if self.hierarchical:
            x2 = relu(self.bn2(self.conv2(x2 + x1)))
            x3 = relu(self.bn3(self.conv3(x3 + x2)))
            x4 = relu(self.bn4(self.conv4(x4 + x3)))
            x5 = relu(self.bn5(self.conv5(x5 + x4)))
            concat = torch.cat([x1, x2, x3, x4, x5], dim=1)
            concat = relu(self.bn6(self.conv6(concat)))
            output = concat + relu(self.bn_residual(self.conv_residual(x0)))
        else:
            output = torch.cat([x1, x2, x3, x4, x5], dim=1)
            output = relu(self.bn6(self.conv6(output)))
        return output


class SimpleDecoder(nn.Module):
    def __init__(self, num_classes, planes):
        super(SimpleDecoder, self).__init__()
        self.last_convolution = nn.Conv2d(planes, num_classes, kernel_size=1, bias=False)

    def forward(self, x, target_size=None):
        x = self.last_convolution(x)
        x = interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class ComplexDecoder(nn.Module):
    def __init__(self, num_classes, sideplanes, planes=256, depthwise=False):
        super(ComplexDecoder, self).__init__()
        self.conv_side = nn.Conv2d(sideplanes, planes, kernel_size=1, bias=False)
        self.bn_side = nn.BatchNorm2d(planes)
        self.conv1 = Conv(planes * 2, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv(planes, planes, kernel_size=3, bias=False, padding=1, depthwise=depthwise)
        self.bn2 = nn.BatchNorm2d(planes)

        self.last_convolution = nn.Conv2d(planes, num_classes, kernel_size=1, bias=False)

    def forward(self, x, x_side, target_size=None):
        x = interpolate(x, size=x_side.shape[-2:], mode='bilinear', align_corners=False)
        x_side = relu(self.bn_side(self.conv_side(x_side)))
        x = torch.cat([x, x_side], dim=1)
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = self.last_convolution(x)
        x = interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class DeepLabV3(nn.Module):
    def __init__(self, encoder, num_classes=19, depthwise=False, decoder=False, hierarchical=False):
        super(DeepLabV3, self).__init__()
        self.encoder = encoder
        self.inplanes = self.encoder.get_dimensions()[-1]
        output_stride = self.encoder.get_ouput_stride()

        self.pyramid = PyramidModule(self.inplanes, 256, depthwise, output_stride)
        self.features_fuser = FeaturesFuser(256, depthwise=depthwise, hierarchical=hierarchical, inplanes=self.inplanes)

        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

        self.complex_decoder = decoder
        if decoder:
            self.decoder = ComplexDecoder(num_classes, self.encoder.get_dimensions()[1],
                                          planes=256, depthwise=depthwise)
        else:
            self.decoder = SimpleDecoder(num_classes, 256)

    def forward(self, x):
        if self.training and x.shape[0] == 1:
            print("Warning ! DeepLabV3's batchnorms may encounter critical errors when training with batch_size of 1 !")
        if self.complex_decoder:
            outputs = self.encoder(x)
            output = outputs[-1]
            output2 = outputs[1]
        else:
            output = self.encoder(x)[-1]
            output2 = None

        x1, x2, x3, x4, x5 = self.pyramid(output)
        output = self.features_fuser(x1, x2, x3, x4, x5, output)

        if self.complex_decoder:
            output = self.decoder(output, output2, target_size=x.shape[-2:])
        else:
            output = self.decoder(output, target_size=x.shape[-2:])

        return output
