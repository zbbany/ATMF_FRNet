#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath
from torch.nn import BatchNorm2d
import copy
import os

eps = 1e-12

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.ModuleList, nn.GELU,nn.Dropout2d, DropPath,  nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)
        self.initialize()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    def initialize(self):
        weight_init(self)

class BaseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.Sequential(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      groups=groups,
                      bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.initialize()

    def forward(self, x):
        return self.basicconv(x)
    def initialize(self):
        weight_init(self)

class CA(nn.Module):
    def __init__(self, channel, ratio=2):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio),
            nn.ReLU(True),
            nn.Linear(channel // ratio, channel),
            nn.Sigmoid()
        )
        self.initialize()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y
    def initialize(self):
        weight_init(self)

class CA1(nn.Module):
    def __init__(self, in_planes):
        super(CA1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    def initialize(self):
        weight_init(self)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)
    def initialize(self):
        weight_init(self)

class DepthFuseNet(nn.Module):
    def __init__(self, inchannels):
        super(DepthFuseNet, self).__init__()
        self.d_linear = nn.Sequential(
            nn.Linear(inchannels, inchannels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(inchannels, inchannels, bias=True),
        )
        self.initialize()

    def forward(self, x):
        x_d1 = self.d_linear(x.mean(dim=2).mean(dim=2)).unsqueeze(dim=2).unsqueeze(dim=3)
        x_f1 = torch.sigmoid(x_d1)
        return x_f1
    def initialize(self):
        weight_init(self)


class smAR(nn.Module):
    def __init__(self, in_channels):
        super(smAR, self).__init__()
        self.ca = CA(in_channels)
        self.sa = SA()
        self.gap = DepthFuseNet(in_channels)
        self.conv = BaseConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.rgb = BaseConv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.initialize()

    def forward(self, rgb):
        rgb_SA = self.sa(rgb)*rgb
        rgb_CA = self.ca(rgb)*rgb
        rgb_M = self.rgb(torch.cat([rgb_SA, rgb_CA], dim=1))
        gap = self.gap(rgb) * rgb + rgb
        rgb_M = rgb_M * gap + gap
        rgb_smAR = self.conv(rgb_M)

        return rgb_smAR
    def initialize(self):
        weight_init(self)


class PSPModule(nn.Module):
    def __init__(self, features, out_features, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )
        self.initialize()

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = BatchNorm2d(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle
    def initialize(self):
        weight_init(self)

class RRGN(nn.Module):
    def __init__(self,inner_channels=128):
        super(RRGN, self).__init__()
        self.side_conv1 = nn.Conv2d(256, inner_channels, kernel_size=3, stride=1, padding=1) #self.side_conv1 = nn.Conv2d(64, inner_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(512, inner_channels, kernel_size=3, stride=1, padding=1) #self.side_conv2 = nn.Conv2d(128, inner_channels, kernel_size=3, stride=1, padding=1)#
        self.side_conv3 = nn.Conv2d(1024, inner_channels, kernel_size=3, stride=1, padding=1) #self.side_conv3 = nn.Conv2d(320, inner_channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(2048, inner_channels, kernel_size=3, stride=1, padding=1) #self.side_conv4 = nn.Conv2d(512, inner_channels, kernel_size=3, stride=1, padding=1)

        self.E1_1 = self._make_layer(inner_channels, inner_channels, 3, 1, 1)
        self.E1_2 = self._make_layer(inner_channels, inner_channels, 1, 1, 0)
        self.E1_3 = self._make_layer(inner_channels, inner_channels, 1, 1, 0)

        self.E2_1 = self._make_layer(inner_channels, inner_channels, 3, 2, 1)
        self.E2_2 = self._make_layer(inner_channels, inner_channels, 3, 1, 1)
        self.E2_3 = self._make_layer(inner_channels, inner_channels, 1, 1, 0)

        self.E3_1 = self._make_layer_2(inner_channels, inner_channels, 3, 2, 1)
        self.E3_2 = self._make_layer(inner_channels, inner_channels, 3, 2, 1)
        self.E3_3 = self._make_layer(inner_channels, inner_channels, 3, 1, 1)
        self.PSP = PSPModule(inner_channels, inner_channels//2)
        self.smAR = smAR(inner_channels)
        self.initialize()

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample(self, x,scale):
        _,_,h,w=x.size()
        return F.interpolate(x, size=(h*scale, w*scale), mode='bilinear', align_corners=True)

    def _make_layer_2(self, num_channels_pre_layer, num_channels_cur_layer,kerner_size,stride,padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
            nn.Conv2d(num_channels_pre_layer,num_channels_cur_layer,kerner_size,stride,padding,bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels_cur_layer, num_channels_cur_layer, kerner_size, stride, padding, bias=False),
            nn.BatchNorm2d(num_channels_cur_layer),
        ))
        return nn.Sequential(*transition_layers)

    def _make_layer(self, num_channels_pre_layer, num_channels_cur_layer, kerner_size, stride, padding):
        transition_layers = []
        transition_layers.append(nn.Sequential(
                nn.Conv2d(num_channels_pre_layer, num_channels_cur_layer, kerner_size, stride, padding, bias=False),
                nn.BatchNorm2d(num_channels_cur_layer),
                nn.ReLU(inplace=True),
        ))
        return nn.Sequential(*transition_layers)

    def forward(self, E4, E3, E2, E1):
        E4, E3, E2, E1 = self.side_conv4(E4), self.side_conv3(E3), self.side_conv2(E2), self.side_conv1(E1)
        E4 = self.smAR(E4)
        E3 = self._upsample_add(E4, E3)
        E2 = self._upsample_add(E4, E2)
        E1 = self._upsample_add(E4, E1)

        E1_1 = self.E1_1(E1)
        E1_2 = self._upsample(self.E1_2(E2), scale=2)
        E1_3 = self._upsample(self.E3_3(E3), scale=4)

        E2_1 = self.E2_1(E1)
        E2_2 = self.E2_2(E2)
        E2_3 = self._upsample(self.E2_3(E3), scale=2)

        E3_1 = self.E3_1(E1)
        E3_2 = self.E3_2(E2)
        E3_3 = self.E3_3(E3)

        E1 = self.PSP(E1_1 + E1_2 + E1_3)
        E2 = self.PSP(E2_1 + E2_2 + E2_3)
        E3 = self.PSP(E3_1 + E3_2 + E3_3)

        return E1, E2, E3
    def initialize(self):
        weight_init(self)

class IntraCLBlock(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(IntraCLBlock, self).__init__()
        self.in_channels = in_channels
        self.c_layer_7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.conv7 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=7, padding=7)
        self.c_layer_5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=5, padding=5)
        self.c_layer_3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=3, padding=3)
        self.c_layer_1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=(1, 1), padding=0)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, padding=1)

        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

        self.ca = CA1(in_channels)
        self.sa = SA()
        self.gap = DepthFuseNet(in_channels)
        self.rgb = BaseConv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.sal_conv = nn.Sequential(
            ConvBnRelu(in_channels, in_channels, 3, padding=1),
            ConvBnRelu(in_channels, in_channels, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()
        self.initialize()

    def forward(self, input):
        x = self.conv1x1(input)

        x1 = self.c_layer_1x1(x)
        x1 = self.conv1(x1)
        x1 = nn.ReLU()(x1)

        x3 = self.c_layer_3x3(x+x1)
        x3 = self.conv3(x3)
        x3 = nn.ReLU()(x3)

        x5 = self.c_layer_5x5(x+x3)
        x5 = self.conv5(x5)
        x5 = nn.ReLU()(x5)

        x7 = self.c_layer_7x7(x+x5)
        x7 = self.conv7(x7)
        x7 = nn.ReLU()(x7)
        out = self.conv3x3(torch.cat([x1, x3, x5, x7], dim=1))
        out = out + input

        rgb_SA = self.sa(input) * input
        rgb_CA = self.ca(input) * input
        rgb_M = self.rgb(torch.cat([rgb_SA, rgb_CA], dim=1))
        gap = self.gap(out) * out + out
        rgb_M = rgb_M * gap + gap
        rgb_smAR = self.sal_conv(rgb_M) + input + out
        return rgb_smAR

    def initialize(self):
        weight_init(self)

class FPN(nn.Module):
    def __init__(self, inner_channels=64, **kwargs):
        super().__init__()
        inplace = True
        self.Intra = IntraCLBlock(inner_channels)
        self.smooth_p = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channels * 3, inner_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=inplace)
        )
        self.initialize()
    def forward(self, x):
        f1, f2, f3 = x

        f3 = self._upsample_add(f1, f3)
        f3 = self.smooth_p(self.Intra(f3))
        f2 = self._upsample_add(f3, f2)
        f2 = self.smooth_p(self.Intra(f2))
        f1 = self._upsample_add(f2, f1)
        f1 = self.smooth_p(self.Intra(f1))
        pose = self._upsample_cat(f1, f2, f3)
        pose = self.conv(pose)

        return f3, f2, f1, pose
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y
    def _upsample_cat(self, p2, p3, p4):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        return torch.cat([p2, p3, p4], dim=1)
    def initialize(self):
        weight_init(self)


class ICON(torch.nn.Module):
    def __init__(self, cfg, model_name='ICON-R'):
        super(ICON, self).__init__()
        self.fpn = FPN()
        self.rrgn = RRGN()
        self.predtrans = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.initialize()
    def forward(self, x, shape=None, name=None):
        x = x.to(torch.float32)
        features = self.encoder(x)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        features = self.rrgn(x1, x2, x3, x4)
        x1, x2, x3, x4 = self.fpn(features)

        if shape is None:
            shape = x.size()[2:]
        pred1 = F.interpolate(self.predtrans(x1), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.predtrans(x2), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.predtrans(x3), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.predtrans(x4), size=shape, mode='bilinear')

        return pred1, pred2, pred3, pred4

