import torch.nn as nn
from models.shufflenet import ShuffleNetV2


class FCN8(nn.Module):
    def __init__(self, encoder, num_classes=19):
        super(FCN8, self).__init__()

        self.encoder = encoder
        encoder_dims = self.encoder.get_dimensions()

        if isinstance(self.encoder, ShuffleNetV2):
            self.score_output = nn.Conv2d(encoder_dims[-1], num_classes, 1)  # Output
            self.score_pool3 = nn.Conv2d(encoder_dims[1], num_classes, 1)  # Pool4
            self.score_pool4 = nn.Conv2d(encoder_dims[2], num_classes, 1)  # Pool3
        else:
            self.score_output = nn.Conv2d(encoder_dims[-1], num_classes, 1)  # Output
            self.score_pool3 = nn.Conv2d(encoder_dims[2], num_classes, 1)  # Pool4
            self.score_pool4 = nn.Conv2d(encoder_dims[3], num_classes, 1)  # Pool3

        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upsample8 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)

    def forward(self, x):
        outputs = self.encoder(x)

        if isinstance(self.encoder, ShuffleNetV2):
            x_output, x_pool3, x_pool4 = outputs[-1], outputs[1], outputs[2]
        else:
            x_output, x_pool3, x_pool4 = outputs[-1], outputs[2], outputs[3]
        x_output = self.score_output(x_output)
        x_pool3 = self.score_pool3(x_pool3)
        x_pool4 = self.score_pool4(x_pool4)

        upsampled_2 = self.upsample2(x_output)
        upsampled_2 = upsampled_2[:, :, 1:-1, 1:-1]

        upsampled_4 = self.upsample4(upsampled_2 + x_pool4)
        upsampled_4 = upsampled_4[:, :, 1:-1, 1:-1]

        upsampled_8 = self.upsample8(upsampled_4 + x_pool3)
        upsampled_8 = upsampled_8[:, :, 4:-4, 4:-4]

        return upsampled_8
