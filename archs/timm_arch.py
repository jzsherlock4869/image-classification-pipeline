import torch
import torch.nn as nn

import timm

class TimmArch(nn.Module):
    """
        Use the timm library to construct arch
        simple modify in this class if necessary 
    """
    def __init__(self, backbone='resnet34', num_classes=10, pretrained=False):
        super(TimmArch, self).__init__()
        self.arch = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        out = self.arch(x)
        return out