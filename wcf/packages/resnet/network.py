from .resnets import resnet18, resnext101_32x8d, resnext50_32x4d, resnet152, resnet101, resnet50, resnet34, \
    wide_resnet50_2, wide_resnet101_2
import torch
from torch import nn

resnets_dict = dict(
    resnet18=resnet18,
    resnet34=resnet34,
    resnet50=resnet50,
    resnet101=resnet101,
    resnet152=resnet152,
    resnext50_32x4d=resnext50_32x4d,
    resnext101_32x8d=resnext101_32x8d,
    wide_resnet50_2=wide_resnet50_2,
    wide_resnet101_2=wide_resnet101_2,
)


class Resnet(nn.Module):
    def __init__(self, num_classes, pretrained=False, type='resnet18'):
        super().__init__()
        model = resnets_dict[type](pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return x
