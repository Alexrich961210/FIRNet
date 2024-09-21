import torch.nn as nn
import math
import torch.nn.init as init
from modules import *

__all__ = ['vgg_small_1w1a']

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
    
class v3conv(nn.Module):

    def __init__(self, inplanes, planes, pooling=False, groups_num=2):
        super(v3conv, self).__init__()

        self.groups_num = groups_num
        self.pooling = pooling
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d((2,2))
        self.downsample = nn.Identity()

        self.binary_conv1 = BinarizeConv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)

        self.shift = LearnableBias(planes)
        self.hardtanh = nn.Hardtanh()

        self.inplanes = inplanes

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        assert c % groups == 0, "Number of channels must be divisible by groups."
    
        x = x.view(b, groups, c // groups, h, w)
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(b, c, h, w)
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(b, -1, h, w)
    
        return x
    
    def forward(self, x):

        res = x
        x_act, x_idle = torch.split(x, [self.inplanes, self.inplanes], dim=1)

        out = self.binary_conv1(x_act)
        if self.pooling == True:
            out = self.pool(out)
            x_idle = self.avgpool(x_idle)
        else:
            x_idle = res

        out = self.bn1(out)
        out = self.hardtanh(out)

        out = torch.cat([out, x_idle], dim=1)
        if self.groups_num > 1:
            out = self.channel_shuffle(out, groups=self.groups_num)

        return out
        
def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class DownBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(DownBlock, self).__init__()

        self.conv = BinarizeConv2d(in_planes*2, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.nonlinear = nn.Hardtanh(inplace=True)


        self.inplanes = in_planes
    
    def forward(self, x):

        residual1, _ = torch.split(x, [self.inplanes, self.inplanes], dim=1)

        out = self.bn(self.conv(x))

        out += residual1
        out = self.nonlinear(out)

        return out
    
class VGG_SMALL_1W1A(nn.Module):
    def __init__(self, num_classes=10, groups_num=2):
        super(VGG_SMALL_1W1A, self).__init__()
        self.groups_num = groups_num
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)

        self.nonlinear = nn.Hardtanh()

        self.layer1 = v3conv(128, 128, pooling=True)

        # self.conv2 = BinarizeConv2d(128*groups_num, 256*groups_num, kernel_size=3, padding=1, groups=groups_num, bias=False)
        # self.bn2 = nn.BatchNorm2d(256*groups_num)
        self.layer2 = v3conv(128, 256)
   
        # self.conv3 = BinarizeConv2d(256*groups_num, 256*groups_num, kernel_size=3, padding=1, groups=groups_num, bias=False)
        # self.bn3 = nn.BatchNorm2d(256*groups_num)
        self.layer3 = v3conv(256, 256, pooling=True)
       
        # self.conv4 = BinarizeConv2d(256*groups_num, 512*groups_num, kernel_size=3, padding=1, groups=groups_num, bias=False)
        # self.bn4 = nn.BatchNorm2d(512*groups_num)
        self.layer4 = v3conv(256, 512)
    
        # self.conv5 = BinarizeConv2d(512*groups_num, 512*groups_num, kernel_size=3, padding=1, groups=groups_num, bias=False)
        # self.bn5 = nn.BatchNorm2d(512*groups_num)
        self.layer5 = v3conv(512, 512, pooling=True)

        self.downblock = DownBlock(512, 512, 1)
  
        self.bn = nn.BatchNorm1d(512*4*4)
        self.fc = nn.Linear(512*4*4, num_classes)
        self._initialize_weights()
        # self.apply(_weights_init)

    def channel_shuffle(self, x, groups):
        b, c, h, w = x.size()
        assert c % groups == 0, "Number of channels must be divisible by groups."
    
        x = x.view(b, groups, c // groups, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, c, h, w)
    
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BinarizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        # x = self.shift(x)
        x = self.nonlinear(x)
        x = torch.cat([x for _ in range(self.groups_num)], dim =1)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)

        x = self.downblock(x)


        x = x.view(x.size(0), -1)
        x = self.bn(x)
        x = self.fc(x)
        return x


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model