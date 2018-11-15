from collections import OrderedDict

import pretrainedmodels
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.v1.config import *
from src.v1.utils import load_checkpoint


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint=None):
    model = ShipClassifier()
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    model.train()
    model.to(DEVICE)
    print("Trainable Parameters: %s" % count_parameters(model))
    return model, state


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ConvBnRelu(nn.Module):
    def __init__(self, in_, out, bn=True):
        super().__init__()
        self.bn = bn
        self.conv = conv3x3(in_, out)
        if self.bn:
            self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class ShipClassifier(nn.Module):

    def __init__(self):
        super(ShipClassifier, self).__init__()

        encoder_outputs = [64, 256, 512, 1024, 2048]

        self.encoder = pretrainedmodels.__dict__['se_resnet50'](pretrained='imagenet')
        layer0_modules = [
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=1,
                                padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2))
        ]

        self.conv1 = nn.Sequential(OrderedDict(layer0_modules))

        # self.conv1 = self.encoder.layer0  # Drop last pooling layer

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(
            ConvBnRelu(encoder_outputs[4], 512),
            ConvBnRelu(512, 512),
            nn.MaxPool2d(kernel_size=2)
        )
        self.logit = nn.Linear(512, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 64 x 64 x 64
        conv2 = self.conv2(conv1)  # 256 x 64 x 64
        conv3 = self.conv3(conv2)  # 512 x 32 x 32
        conv4 = self.conv4(conv3)  # 1024 x 16 x 16
        conv5 = self.conv5(conv4)  # 2048 x 8 x 8
        center = self.center(conv5)
        pooled = F.adaptive_max_pool2d(center, 1)
        pooled = pooled.view(inputs.size(0), -1)
        logit = self.logit(F.dropout(pooled, p=0.5, training=self.training))
        return logit

    def freeze_encoder(self):
        freeze_params(self.conv1.parameters())
        freeze_params(self.conv2.parameters())
        freeze_params(self.conv3.parameters())
        freeze_params(self.conv4.parameters())
        freeze_params(self.conv5.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.trainable_params(), lr=1e-3, weight_decay=0.0001)
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(mode='max', optimizer=optimizer, min_lr=1e-3,
                                      patience=8, factor=0.5, verbose=True)
        return scheduler

    def get_loss(self):
        return torch.nn.BCEWithLogitsLoss()


def freeze_params(params):
    for param in params:
        param.requires_grad = False


if __name__ == '__main__':
    model = ShipClassifier()
    print("Testing model: Params: %d" % count_parameters(model))
    x = torch.empty((1, 3, 224, 224))
    y = model(x)
    print(x.size(), '-->', y.size())
