import pretrainedmodels
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models

from src.utils import load_checkpoint
from src.v2.config import LOAD_CHECKPOINT, DEVICE
from src.v2.loss import FocalLovasz, MixedLoss
from src.v2.config import segmentator as PARAM


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


class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes=1, use_aux=True):
        super(PSPNet, self).__init__()
        self.use_aux = use_aux
        resnet = pretrainedmodels.__dict__['resnet34'](pretrained='imagenet')

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.ppm = _PyramidPoolingModule(512, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(2048 + 512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(256, num_classes, kernel_size=1)
            initialize_weights(self.aux_logits)

        initialize_weights(self.ppm, self.final)

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')

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


def load_model(checkpoint=None):
    model = PSPNet()
    print("PSPNet is loading!")
    state = {'epoch': 0, 'lb_acc': 0}
    if LOAD_CHECKPOINT:
        state = load_checkpoint(model, checkpoint)

    # model.freeze_encoder()
    model.train()
    model.to(DEVICE)
    print("Trainable Parameters: %s" % count_parameters(model))
    return model, state


if __name__ == '__main__':
    model = PSPNet()
    print("Testing model: Params: %d" % count_parameters(model))
    x = torch.empty((2, 3, 256, 256))
    y = model(x)
    print(x.size(), '-->', y.size())
