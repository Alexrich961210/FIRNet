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
from modules import *
from torch.autograd import Function,Variable

__all__ = ['resnet20_1w1a']

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
# class BasicBlock_1w1a(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock_1w1a, self).__init__()
#         self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         pad = 0 if planes == self.expansion*in_planes else planes // 4
#         if stride != 1 or in_planes != planes:
#             self.shortcut = nn.Sequential(
#                         nn.AvgPool2d((2,2)), 
#                         LambdaLayer(lambda x:
#                         F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))


#     def forward(self, x):
#         out = self.bn1(self.conv1(x))
#         out += self.shortcut(x) 
#         out = F.hardtanh(out, inplace=True)
#         x1 = out
#         out = self.bn2(self.conv2(out))
#         out += x1 
#         out = F.hardtanh(out, inplace=True)
#         return out  
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.in_planes = inplanes
        self.planes = planes

        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        # self.nonlinear1 = nn.ReLU(inplace=True)
        # self.nonlinear2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        pad = 0 if planes == inplanes else planes // 4
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

        self.pool = nn.AvgPool2d((2,2))

        self.move0 = LearnableBias(planes)
        self.move1 = LearnableBias(planes)
    
    def channel_shuffle(self, x, groups=2):
        b, c, h, w = x.size()
        assert c % groups == 0, "Number of channels must be divisible by groups."
    
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)
    
        return x

    def forward(self, x):
        res = x

        x_act, x_idle = torch.split(x, [self.in_planes, self.in_planes], dim=1)
        residual = x_act
        
        out = self.bn1(self.conv1(x_act))         
        if self.downsample is not None:
            residual = self.downsample(residual)
            x_idle = self.pool(res)
            # x_idle = residual
            # x_idle = self.downsample(x_idle)

        out += residual

        out = self.nonlinear1(out)

        x_idle = self.move0(x_idle)
        out = torch.cat([out, x_idle], dim=1)
        out = self.channel_shuffle(out)


        x_act2, x_idle2 = torch.split(out, [self.planes, self.planes], dim=1)
        residual = x_act2
  
        out = self.conv2(x_act2)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear2(out)

        x_idle2 = self.move1(x_idle2)
        out = torch.cat([out, x_idle2], dim=1)
        out = self.channel_shuffle(out)

        return out
class BasicBlock_ablation1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_ablation1, self).__init__()
        self.in_planes = inplanes
        self.planes = planes

        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.nonlinear2 = nn.Hardtanh(inplace=True)

        self.downsample = downsample
        self.stride = stride

        pad = 0 if planes == inplanes else planes // 4
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))

        self.pool = nn.AvgPool2d((2,2))

        self.move0 = LearnableBias(planes)
        self.move1 = LearnableBias(planes)
    
    def channel_shuffle(self, x, groups=2):
        b, c, h, w = x.size()
        assert c % groups == 0, "Number of channels must be divisible by groups."
    
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)
    
        return x

    def forward(self, x):
        res = x

        x_act, x_idle = torch.split(x, [self.in_planes, self.in_planes], dim=1)
        residual = x_act
        
        out = self.bn1(self.conv1(x_act))         
        if self.downsample is not None:
            residual = self.downsample(residual)

            out += residual

            out = self.nonlinear1(out)

            out = torch.cat([out, out], dim=1)
        else:
            out += residual

            out = self.nonlinear1(out)

            x_idle = self.move0(x_idle)
            out = torch.cat([out, x_idle], dim=1)
            out = self.channel_shuffle(out)



        x_act2, x_idle2 = torch.split(out, [self.planes, self.planes], dim=1)
        residual = x_act2
  
        out = self.conv2(x_act2)
        out = self.bn2(out)

        out += residual
        out = self.nonlinear2(out)

        x_idle2 = self.move1(x_idle2)
        out = torch.cat([out, x_idle2], dim=1)
        out = self.channel_shuffle(out)

        return out
class DownBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(DownBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes

        self.conv = BinarizeConv2d(in_planes*2, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv = nn.Conv2d(in_planes*2, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.nonlinear = nn.Hardtanh(inplace=True)
    
    def forward(self, x):

        # feature fusion conv with 3x3 kernel size)
        residual1, residual2 = torch.split(x, [self.planes, self.planes], dim=1)
        out = self.bn(self.conv(x))
        out += residual1
        out = self.nonlinear(out)

        # out = residual1 + residual2

        # # out = self.bn(self.conv(x))
        # # out = self.nonlinear(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.downblock = DownBlock(64, 64)
        # self.fc = nn.Linear(128, 64)

        self.bn2 = nn.BatchNorm1d(64)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)), inplace=True)
        out = torch.cat([out for _ in range(2)], dim =1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.downblock(out)
        out = F.avg_pool2d(out, out.size()[3])

        out = out.view(out.size(0), -1) 
        out = self.bn2(out)
        out = self.linear(out)

        return out


def resnet20_1w1a(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3],num_classes=num_classes)

