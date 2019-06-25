##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict

class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(PixelAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(key_channels))]))
        self.f_query = self.f_key

    def forward_att(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map

    def forward(self, x):
        raise NotImplementedError


class SelfAttentionBlock_(PixelAttentionBlock_):
    def __init__(self, in_channels, key_channels, value_channels, out_channels, scale=1):
        super(SelfAttentionBlock_, self).__init__(in_channels, key_channels)
        self.scale = scale
        self.value_channels = value_channels
        self.out_channels = out_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_value = nn.Conv2d(in_channels, value_channels, kernel_size=1, stride=1, padding=0)
        self.f_output = nn.Conv2d(value_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.parameter_initialization()

    def parameter_initialization(self):
        nn.init.constant_(self.f_output.weight, 0)
        nn.init.constant_(self.f_output.bias, 0)

    def forward(self, x):
        batch_size, c, h, w = x.size()
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.forward_att(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.f_output(context)
        if self.scale > 1:
            context = F.upsample(context, size=(h, w), mode='bilinear', align_corners=True)
        return [context, sim_map]

class SADecoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=512, dilations=(6, 12, 18), scale=1):
        super(SADecoder, self).__init__()
        d1, d2, d3 = dilations
        self.saconv = nn.Sequential(OrderedDict([
            ('conv',   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('saconv', SelfAttentionBlock_(out_channels, out_channels//2, out_channels, out_channels, scale))]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,  dilation=1,  bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels))]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d1, dilation=d1, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels))]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d2, dilation=d2, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels))]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d3, dilation=d3, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels))]))

        self.merge = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('dropout', nn.Dropout2d(0.1))]))
            
    def get_sim_map(self):
        return self.sim_map

    def forward(self, x):
        feat1, self.sim_map = self.saconv(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x) 
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        out = self.merge(out)
        return out