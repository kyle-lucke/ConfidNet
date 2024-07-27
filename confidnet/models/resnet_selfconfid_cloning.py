# 

##########################################################################################
#
# Taken from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py 
#
##########################################################################################

'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from confidnet.models.model import AbstractModel
from confident.models.resnet import resnet32
from confident.models.resnet_selfconfid_classic import 

class ResNetSelfConfidCloning(AbstractModel):    
    def __init__(self, config_args, device, block, num_blocks):
        super().__init__(config_args, device)

        self.pred_network = resnet32(config_args, device)
        config_args["data"]["num_classes"] = 1
        self.uncertainty_network = resnet32_selfconfid_classic(config_args, device)

    def forward(self, x):
      pred = self.pred_network(x)
      _, uncertainty = self.uncertainty_network(x)
      return pred, uncertainty

def resnet32_selfconfid_cloning(config_args, device):
    raise NotImplementedError()
    model = ResNetSelfConfidCloning(config_args, device, BasicBlock, [5, 5, 5]) 
    return model

if __name__ == "__main__":

    resnet = resnet32(10)

    fake_inp = torch.rand(2, 3, 32, 32)

    res = resnet(fake_inp)

    print(res.shape)
