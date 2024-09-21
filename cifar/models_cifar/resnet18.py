'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from modules import *


__all__ =['resnet18A_1w1a','resnet18B_1w1a','resnet18C_1w1a','resnet18_1w1a']

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

class BasicBlock_1w1a(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_1w1a, self).__init__()
        self.in_planes = inplanes
        self.planes = planes

        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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
            x_idle = self.pool(res)

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

        self.move0 = LearnableBias(planes)
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)


        self.conv2 = BinarizeConv2d(planes*2, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.nonlinear2 = nn.Hardtanh(inplace=True)

    
    def channel_shuffle(self, x, groups=2):
        b, c, h, w = x.size()
        assert c % groups == 0, "Number of channels must be divisible by groups."
    
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)
    
        return x
    
    def forward(self, x):

        x_act, x_idle = torch.split(x, [self.in_planes, self.in_planes], dim=1)

        residual = x_act
        
        out = self.bn1(self.conv1(x_act))
        out += residual
        out = self.nonlinear1(out)

        x_idle = self.move0(x_idle)
        out = torch.cat([out, x_idle], dim=1)
        out = self.channel_shuffle(out)

        # feature fusion conv with 3x3 kernel size
        residual1, _ = torch.split(out, [self.planes, self.planes], dim=1)

        out = self.bn2(self.conv2(out))

        out = out + residual1
        out = self.nonlinear2(out)

        return out

class Bottleneck_1w1a(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_1w1a, self).__init__()
        self.conv1 = BinarizeConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                BinarizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.hardtanh(self.bn1(self.conv1(x)))
        out = F.hardtanh(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.hardtanh(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channel, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channel[0]

        self.conv1 = nn.Conv2d(3, num_channel[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channel[0])
        self.layer1 = self._make_layer(block, num_channel[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channel[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channel[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channel[3], num_blocks[3], stride=2)
        
        self.downblock = DownBlock(512, 512)
        self.linear = nn.Linear(num_channel[3]*block.expansion, num_classes)
        self.bn2 = nn.BatchNorm1d(num_channel[3]*block.expansion)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = torch.cat([out for _ in range(2)], dim =1)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.downblock(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.linear(out)
        return out 


def resnet18A_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,32,64,128],**kwargs)

def resnet18B_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[32,64,128,256],**kwargs)

def resnet18C_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,64,128,256],**kwargs)

def resnet18_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [2,2,2,2],[64,128,256,512],**kwargs)

def resnet34_1w1a(**kwargs):
    return ResNet(BasicBlock_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet50_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,6,3],[64,128,256,512],**kwargs)

def resnet101_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,4,23,3],[64,128,256,512],**kwargs)

def resnet152_1w1a(**kwargs):
    return ResNet(Bottleneck_1w1a, [3,8,36,3],[64,128,256,512],**kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'linear' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
