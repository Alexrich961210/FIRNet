import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from modules import *

BN = None

__all__ = ['resnet18_1w1a', 'resnet34_1w1a']


def conv3x3Binary(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)
def conv1x1Binary(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    padding=0, bias=False)
class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
class channel_w(nn.Module):
    def __init__(self,out_ch):
        super(channel_w, self).__init__()
        self.w1 =torch.nn.Parameter(torch.rand(1,out_ch,1,1)*0.001,requires_grad=True)

    def forward(self,x):
        out = self.w1 * x
        return out
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.in_planes = inplanes
        self.planes = planes

        self.conv1 = conv3x3Binary(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        # self.nonlinear1 = nn.PReLU(planes)
        self.conv2 = conv3x3Binary(planes, planes)
        self.bn2 = BN(planes)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        # self.nonlinear2 = nn.PReLU(planes)
        self.downsample = downsample
        self.stride = stride

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

        self.nonlinear = nn.Hardtanh(inplace=True)

        # self.conv = conv1x1Binary(planes*2, planes)
        self.conv = conv3x3Binary(planes*2, planes)
        self.bn = BN(planes)
    
    def forward(self, x):
        # feature fusion conv with 3x3 kernel size
        residual1, _ = torch.split(x, [self.planes, self.planes], dim=1)

        out = self.bn(self.conv(x))

        out = out + residual1
        out = self.nonlinear(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, 
                 avg_down=False, bypass_last_bn=False,
                 bn_group_size=1,
                 bn_group=None,
                 bn_sync_stats=False,
                 use_sync_bn=True):

        global BN, bypass_bn_weight_list

        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []


        self.inplanes = 64
        super(ResNet, self).__init__()

        self.avg_down = avg_down
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.nonlinear = nn.Hardtanh(inplace=True)
        # self.nonlinear = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.downblock = DownBlock(512, 512)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1e-8)
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear(x) 
        x = self.maxpool(x)

        x = torch.cat([x for _ in range(2)], dim =1)    

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.downblock(x)
        # x = torch.

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.fc(x)

        return x


def resnet18_1w1a(**kwargs):
    """Constructs a ResNet-18 model. """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34_1w1a(**kwargs):
    """Constructs a ResNet-34 model. """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda name: 'conv' in name or 'fc' in name, [name[0] for name in list(net.named_modules())]))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()