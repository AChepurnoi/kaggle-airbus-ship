import pretrainedmodels
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from src.utils import load_checkpoint

from src.v2.config import segmentator as PARAM
from src.v2.loss import MixedLoss, FocalLovasz


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
class _GlobalConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()
        self.conv_l1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Module):
    def __init__(self, num_classes, input_size, pretrained=True):
        super(GCN, self).__init__()
        self.input_size = input_size
        resnet = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')
        # encoder_outputs = [64, 256, 512, 1024, 2048]
        encoder_outputs = [64, 64, 128, 256, 512]

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(encoder_outputs[4], num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(encoder_outputs[3], num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(encoder_outputs[2], num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(encoder_outputs[1], num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
                           self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

    def forward(self, x):
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        fs1 = self.brm5(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)  # 32
        fs2 = self.brm6(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)  # 64
        fs3 = self.brm7(F.upsample_bilinear(fs2, fm1.size()[2:]) + gcfm4)  # 128
        fs4 = self.brm8(F.upsample_bilinear(fs3, fm0.size()[2:]))  # 256
        out = self.brm9(F.upsample_bilinear(fs4, self.input_size))  # 512

        return out

    def freeze_encoder(self):
        freeze_params(self.layer0.parameters())
        freeze_params(self.layer1.parameters())
        freeze_params(self.layer2.parameters())
        freeze_params(self.layer3.parameters())
        freeze_params(self.layer4.parameters())

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def get_optimizer(self):
        # optimizer = torch.optim.Adam(self.trainable_params(), lr=PARAM['lr'], weight_decay=PARAM['L2'])
        optimizer = torch.optim.Adam(self.trainable_params(), lr=PARAM['lr'], weight_decay=PARAM['L2'])

        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = ReduceLROnPlateau(mode='max', optimizer=optimizer, min_lr=1e-3,
                                      patience=8, factor=0.5, verbose=True)
        return scheduler

    def get_loss(self):
        # return torch.nn.BCEWithLogitsLoss()
        # return MixedLoss(10.0, 2.0)
        return FocalLovasz(2.0)


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


from src.v2.config import *


def load_model(checkpoint=None):
    model = GCN(1, 768)
    print("GCN is loading!")
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    model.train()
    model.to(DEVICE)
    print("Trainable Parameters: %s" % count_parameters(model))
    return model, state


if __name__ == '__main__':
    model = GCN(1, 256)
    print("Testing model: Params: %d" % count_parameters(model))
    x = torch.empty((2, 3, 256, 256))
    y = model(x)
    print(x.size(), '-->', y.size())
