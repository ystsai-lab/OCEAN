import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from model.resNet import ResNet18, ResNet34
from model.sa import SANet

class protoNet_ResNet18_4layer_SA(nn.Module):
    def __init__(self, isPretrained=True, groups=64):
        super(protoNet_ResNet18_4layer_SA, self).__init__()
        self.resNet = ResNet18(isPretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.sa_net = SANet(512, groups=groups)

    def forward_step1(self, x):
        feature_maps = self.resNet(x)
        # 4 layers do average pooling
        output1 = self.avgpool(feature_maps[0])
        output2 = self.avgpool(feature_maps[1])
        output3 = self.avgpool(feature_maps[2])
        output4 = self.avgpool(feature_maps[3])

        return output1, output2, output3, output4
    
    def forward_step2(self, x, proto):
        x_feature_maps = self.resNet(x)
        proto_feature_maps = self.resNet(proto)

        

class protoNet_ResNet18_(nn.Module):
    def __init__(self, isPretrained=True):
        super(protoNet_ResNet18_, self).__init__()
        self.resNet = ResNet18(isPretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_maps = self.resNet(x)
        output = self.avgpool(feature_maps[3])

        return output.view(output.size(0), -1)
    

class protoNet_ResNet34(nn.Module):
    def __init__(self, isPretrained=True):
        super(protoNet_ResNet34, self).__init__()
        self.resNet = ResNet34(isPretrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_maps = self.resNet(x)
        output = self.avgpool(feature_maps[3])

        return output.view(output.size(0), -1)
    
class protoNet_ResNet18_SA(nn.Module):
    def __init__(self, isPretrained=True, groups=64):
        super(protoNet_ResNet18_SA, self).__init__()
        self.resNet = ResNet18(isPretrained)
        self.sa_net = SANet(512, groups=groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_maps = self.resNet(x)
        output = self.sa_net(feature_maps[3])
        output = self.avgpool(output)

        return output.view(output.size(0), -1)
    
