import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import OrderedDict

class GraphConvBlock_(nn.Module):
    def __init__(self, node, channel):
        super(GraphConvBlock_, self).__init__()
        self.Ag = Parameter(torch.rand(node, node))
        self.conv1 = nn.Conv1d(node, node, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channel, channel, kernel_size=1, stride=1, padding=0)
        
    def get_adjecent_matrix(self):
        return self.Ag

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
        self.conv_reduce = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_B = nn.Conv2d(in_channels, node, kernel_size=1, stride=1, padding=0)
        self.GraphConvs = nn.Sequential(OrderedDict([
            ('gconv{:d}'.format(i), GraphConvBlock_(node, mid_channels)) for i in range(n)
        ]))
        self.conv_extend = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def get_map(self):
        map_list = [self.B]
        for i in range(self.n):
            map_list.append(self.GraphConvs[i].get_adjacent_matrix())
        return map_list

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
        B = self.conv_B(x).view(b, -1, h * w)
        self.B = B
        x = self.conv_reduce(x).view(b, -1, h*w).permute(0, 2, 1)
        V = torch.matmul(B, x)

        # Graph Convolution Update
        Z = self.GraphConvs(V)

        # Interaction Space to Coordinate Space
        y = torch.matmul(Z.permute(0, 2, 1), B)
        y = self.conv_extend(y.view(b, -1, h, w))
        return skip_x + y


class GCDecoder(nn.Module):
    def __init__(self, in_channels=2048, node=128, n=1):
        super(GCDecoder, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1, padding=0)
        self.gconv = GloRe2d(in_channels, in_channels//2, in_channels, node, n)
        self.merge = nn.Sequential(
            nn.Dropout2d(0.5, inplace=True),
            nn.Conv2d(in_channels+in_channels//4, in_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5, inplace=False),
        )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.gconv(x)
        x = torch.cat((x1, x2), 1)
        x = self.merge(x)
        return x

    

