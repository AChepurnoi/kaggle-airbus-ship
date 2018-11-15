from collections import OrderedDict

import pretrainedmodels
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.v1.config import *
from src.v1.utils import load_checkpoint
from src.v1.config import segmentator as PARAM


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint=None):
    model = UnetModel()
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    model.train()
    model.to(DEVICE)
    print("Trainable Parameters: %s" % count_parameters(model))
    return model, state


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


class unetUp(nn.Module):
    def __init__(self, in_, skip_, middle_, out_):
        super(unetUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_, in_, kernel_size=4, stride=2, padding=1)
        self.conv1 = ConvBnRelu(in_ + skip_, middle_)
        self.conv2 = ConvBnRelu(middle_, out_)
        # self.scse = SELayer(out_)

    def forward(self, inputs, skip=None):
        outputs2 = self.up(inputs)
        if skip is not None:
            offset = outputs2.size()[2] - skip.size()[2]
            padding = 2 * [offset // 2, offset // 2]
            outputs1 = F.pad(skip, padding, mode='replicate')
            catted = torch.cat([outputs1, outputs2], 1)
        else:
            catted = outputs2

        x = self.conv1(catted)
        x = self.conv2(x)
        # x = self.scse(x)
        return x


class UnetDsv(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))

    def forward(self, input):
        return self.dsv(input)


class UnetModel(nn.Module):

    def __init__(self, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(UnetModel, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm

        encoder_outputs = [64, 256, 512, 1024, 2048]
        # encoder_outputs = [64, 64, 128, 256, 512]

        filter_param = 64

        decoder_sizes = [filter_param, filter_param, filter_param, filter_param, filter_param, filter_param]
        middles = [64, 64, 128, 256, 256, 256]

        self.encoder = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')

        layer0_modules = [
            ('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]

        self.conv1 = nn.Sequential(OrderedDict(layer0_modules))

        # self.conv1 = self.encoder.layer0  # Drop last pooling layer

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBnRelu(encoder_outputs[4], 512),
            ConvBnRelu(512, 512)
        )

        self.up_concat5 = unetUp(in_=512, skip_=encoder_outputs[4],
                                 middle_=middles[4], out_=decoder_sizes[4])

        self.up_concat4 = unetUp(in_=decoder_sizes[4], skip_=encoder_outputs[3],
                                 middle_=middles[3], out_=decoder_sizes[3])

        self.up_concat3 = unetUp(in_=decoder_sizes[3], skip_=encoder_outputs[2],
                                 middle_=middles[2], out_=decoder_sizes[2])

        self.up_concat2 = unetUp(in_=decoder_sizes[2], skip_=encoder_outputs[1],
                                 middle_=middles[1], out_=decoder_sizes[1])

        self.up_concat1 = unetUp(in_=decoder_sizes[1], skip_=encoder_outputs[0],
                                 middle_=middles[0], out_=decoder_sizes[0])

        self.final = nn.Conv2d(sum(decoder_sizes), 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(conv5)

        up5 = self.up_concat5(center, conv5)

        up4 = self.up_concat4(up5, conv4)

        up3 = self.up_concat3(up4, conv3)

        up2 = self.up_concat2(up3, conv2)

        up1 = self.up_concat1(up2, conv1)

        h5 = F.upsample(up5, scale_factor=16, mode='bilinear')
        h4 = F.upsample(up4, scale_factor=8, mode='bilinear')
        h3 = F.upsample(up3, scale_factor=4, mode='bilinear')
        h2 = F.upsample(up2, scale_factor=2, mode='bilinear')

        f = F.dropout2d(torch.cat([up1, h2, h3, h4, h5], dim=1), p=0.1, training=self.training)

        final = self.final(f)

        return final

    def freeze_encoder(self):
        freeze_params(self.conv1.parameters())
        freeze_params(self.conv2.parameters())
        freeze_params(self.conv3.parameters())
        freeze_params(self.conv4.parameters())
        freeze_params(self.conv5.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.trainable_params(), lr=PARAM['lr'], weight_decay=PARAM['L2'])
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(mode='max', optimizer=optimizer, min_lr=1e-3,
                                      patience=8, factor=0.5, verbose=True)
        return scheduler

    def get_loss(self):
        return torch.nn.BCELoss()
        # return BCEDiceLoss()

def freeze_params(params):
    for param in params:
        param.requires_grad = False


if __name__ == '__main__':
    model = UnetModel()
    print("Testing model: Params: %d" % count_parameters(model))
    x = torch.empty((1, 3, 224, 224))
    y = model(x)
    print(x.size(), '-->', y.size())
