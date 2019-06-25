##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict

class GraphConvBlock_(nn.Module):
    def __init__(self, node, channel):
        super(GraphConvBlock_, self).__init__()
        self.Ag = Parameter(torch.zeros(node, node))
        self.conv1 = nn.Conv1d(node, node, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        """
        :param x: shape [batch, node, channel]
        :return: x shape [batch, node, channel]
        """
        x = torch.matmul(x.permute(0, 2, 1), 1 - self.Ag)
        x = x.permute(0, 2, 1)
        x = self.conv1(x).permute(0, 2, 1)
        x = self.conv2(x).permute(0, 2, 1)
        return x

class GloRe2d(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, node, n):
        super(GloRe2d, self).__init__()
        self.n = n
        self.conv_reduce = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(mid_channels))]))
        self.conv_B = nn.Conv2d(in_channels, node, kernel_size=1, stride=1, padding=0, bias=False)
        self.GraphConvs = GraphConvBlock_(node, mid_channels)
        self.conv_extend = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            ('bn',   nn.BatchNorm2d(out_channels))]))
        self.alpha = Parameter(torch.tensor(0.1))

    def get_adjecent_matrix(self):
        """
        Return
        ------
        shape [batch_size, node, node]
        """
        return self.GraphConvs.Ag

    def get_projection_matrix(self):
        """
        Return
        ------
        shape [batch_size, node, h, w]
        """
        return self.B

    def forward(self, x):
        """
        :param x: shape [batch, c, h, w]
                B: shape [batch, node, h*w]
                V: shape [batch, node, c']
        :return: y: shape [batch, h, w, c]
        """
        skip_x = x
        # Coordinate Space to Interaction Space
        b, _, h, w = x.shape
        self.B = self.conv_B(x)
        B = self.B.view(b, -1, h * w)
        x = self.conv_reduce(x).view(b, -1, h*w).permute(0, 2, 1)
        V = torch.matmul(B, x)

        # Graph Convolution Update
        Z = self.GraphConvs(V)

        # Interaction Space to Coordinate Space
        y = torch.matmul(Z.permute(0, 2, 1), B)
        y = self.conv_extend(y.view(b, -1, h, w))
        return skip_x + self.alpha * y

class GCDecoder(nn.Module):
    def __init__(self, in_channels=2048, out_channels=512, node=128, n=1):
        super(GCDecoder, self).__init__()
        self.gconv = nn.Sequential(OrderedDict([
            ('conv',   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('gconv',  GloRe2d(out_channels, out_channels//2, out_channels, node, n))]))
        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0,  dilation=1,  bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels)),]))
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d1, dilation=d1, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels)),]))
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d2, dilation=d2, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels)),]))
        self.conv5 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d3, dilation=d3, bias=False)),
            ('bn',   nn.BatchNorm2d(out_channels)),]))

        self.merge = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, padding=0, dilation=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('dropout', nn.Dropout2d(0.1))]))

    def forward(self, x):
        feat1, = self.gconv(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x) 
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        out = self.merge(out)
        return x

    

