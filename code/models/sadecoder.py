import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict

class SelfAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels):
        super(SelfAttentionBlock_, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(True)
        )
        self.parameter_initialization()
        self.f_query = self.f_key

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        return sim_map


class PixelAttentionBlock_(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels, scale=1):
        super(PixelAttentionBlock_, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if scale > 1:
            self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_value = nn.Sequential(
            nn.Conv2d(in_channels, value_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(value_channels, value_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.pixel_att = SelfAttentionBlock_(in_channels, key_channels)
        self.parameter_initialization()

    def parameter_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        sim_map = self.pixel_att(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1).permute(0, 2, 1)
        context = torch.matmul(sim_map, value).permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w),
                                 mode='bilinear', align_corners=True)
        return [context, sim_map]


class SADecoder(nn.Module):
    def __init__(self, in_channels=2048, key_channels=512, value_channels=2048):
        super(SADecoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0)
        self.saconv = PixelAttentionBlock_(in_channels, key_channels, value_channels)
        self.merge = nn.Sequential(
            nn.Dropout2d(0.5, inplace=True),
            nn.Conv2d(in_channels+in_channels//4, in_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2, sim_map = self.saconv(x)
        x = torch.cat((x1, x2), 1)
        x = self.merge(x)
        return x