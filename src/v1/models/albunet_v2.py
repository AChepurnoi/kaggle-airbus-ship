"""The network definition that was used for a second place solution at the DeepGlobe Building Detection challenge."""
import pretrainedmodels
import torch
import torchvision
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import Sequential
from collections import OrderedDict
from src.v1.config import segmentator as PARAM

from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.v1.loss import MixedLoss, LovaszSymLoss, FocalLoss, FocalLovasz
from src.utils import load_checkpoint
from torch.nn import functional as F


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvBnRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvBnRelu, self).__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            """
                Paramaters for Deconvolution were chosen to avoid artifacts, following
                link https://distill.pub/2016/deconv-checkerboard/
            """

            self.block = nn.Sequential(
                ConvBnRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                                   padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                ConvBnRelu(in_channels, middle_channels),
                ConvBnRelu(middle_channels, out_channels),
            )

    def forward(self, x):
        return self.block(x)


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, pretrained=True, is_deconv=True):
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        # self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        num_filters = 32

        self.relu = nn.ReLU(inplace=True)

        self.encoder = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        encoder_outputs = [64, 256, 512, 1024, 2048]
        # encoder_outputs = [64, 64, 128, 256, 512]
        # decoder_sizes = [num_filters, num_filters, num_filters, num_filters, num_filters, num_filters]
        # middles = [64, 64, 128, 256, 256, 256]

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(encoder_outputs[4], num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(encoder_outputs[4] + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec4 = DecoderBlockV2(encoder_outputs[3] + num_filters * 8, num_filters * 8 * 2, num_filters * 8,
                                   is_deconv)
        self.dec3 = DecoderBlockV2(encoder_outputs[2] + num_filters * 8, num_filters * 4 * 2, num_filters * 2,
                                   is_deconv)
        self.dec2 = DecoderBlockV2(encoder_outputs[1] + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2,
                                   is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvBnRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters * 8 * 2 +
                               num_filters * 2 * 2 +
                               num_filters * 2 +
                               num_filters +
                               num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        h5 = F.upsample(dec5, scale_factor=16, mode='bilinear')
        h4 = F.upsample(dec4, scale_factor=8, mode='bilinear')
        h3 = F.upsample(dec3, scale_factor=4, mode='bilinear')
        h2 = F.upsample(dec2, scale_factor=2, mode='bilinear')

        f = F.dropout2d(torch.cat([dec0, dec1, h2, h3, h4, h5], dim=1), p=0.1, training=self.training)
        x_out = self.final(f)

        return x_out

    def freeze_encoder(self):
        freeze_params(self.conv1.parameters())
        freeze_params(self.conv2.parameters())
        freeze_params(self.conv3.parameters())
        freeze_params(self.conv4.parameters())
        freeze_params(self.conv5.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_optimizer(self):
        # optimizer = torch.optim.Adam(self.trainable_params(), lr=PARAM['lr'], weight_decay=PARAM['L2'])
        optimizer = torch.optim.Adam([
            {'params': self.encoder_params(), 'lr': PARAM['encoder_lr']},
            {'params': self.decoder_params()}],
            lr=PARAM['lr'], weight_decay=PARAM['L2'])

        return optimizer

    def encoder_params(self):
        encoder_params = (self.conv1.parameters(),
                          self.conv2.parameters(),
                          self.conv3.parameters(),
                          self.conv4.parameters(),
                          self.conv5.parameters())
        return self._flatmap_params(encoder_params)

    def decoder_params(self):
        decoder_params = (self.center.parameters(),
                          self.dec5.parameters(),
                          self.dec4.parameters(),
                          self.dec3.parameters(),
                          self.dec2.parameters(),
                          self.dec1.parameters(),
                          self.dec0.parameters(),
                          self.final.parameters())
        return self._flatmap_params(decoder_params)

    def _flatmap_params(self, encoder_params):
        trainable_params = []
        for params in encoder_params:
            trainable = [p for p in params if p.requires_grad]
            trainable_params.extend(trainable)
        return trainable_params

    def get_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(mode='max', optimizer=optimizer, min_lr=1e-3,
                                      patience=8, factor=0.5, verbose=True)
        return scheduler

    def get_loss(self):
        # return torch.nn.BCELoss()
        # return MixedLoss(10.0, 2.0)
        return FocalLovasz(2.0)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from src.config import *


def load_model(checkpoint=None):
    model = AlbuNet()
    print("Albunet v2 is loading!")
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    model.train()
    model.to(DEVICE)
    print("Trainable Parameters: %s" % count_parameters(model))
    return model, state


if __name__ == '__main__':
    model = AlbuNet()
    print("Testing model: Params: %d" % count_parameters(model))
    x = torch.empty((1, 3, 256, 256))
    y = model(x)
    print(x.size(), '-->', y.size())
