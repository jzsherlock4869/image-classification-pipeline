# models from timm

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import local_timm as timm
from local_timm.models.layers import Conv2dSame
from torch import nn

class BasicClasModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_arch = config.model_arch
        ckpt = config.ckpt
        num_class = config.num_class
        num_chs = config.num_input_chs
        if not ckpt == '':
            self.model = timm.create_model(model_arch, pretrained=False, checkpoint_path=ckpt, in_chans=num_chs)
        else:
            self.model = timm.create_model(model_arch, pretrained=False, in_chans=num_chs)
        
        print(self.model)
        #self.model.conv_stem = Conv2dSame(num_chs, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, num_class)

    def forward(self, x):
        x = self.model(x)
        return x

