import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FeatureExtractor(nn.Module):
    def __init__(self, NNModel, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.NNModel = NNModel
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.NNModel._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x.data.cpu().numpy())
                print('{:20}->'.format(name))
            else:
                print(name)
        return x, outputs


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Reshape(nn.Module):
    # input the dimensions except the batch_size dimension
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.permute(self.shape)


class ImageContextBlock(nn.Module):
    def __init__(self, height, width, in_channels, out_channels):
        super(ImageContextBlock, self).__init__()
        self.GlobalPooling = nn.Sequential(
            nn.AvgPool2d((height, width), padding=0),
            nn.Dropout2d(0.5, inplace=True),
            Reshape(in_channels),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            Reshape(out_channels, 1, 1),
        )
        self.upsample = nn.Upsample(size=(height, width),
                                    mode='bilinear', align_corners=True)

    def L2_norm(self, x):
        y = x / torch.norm(x, 2)
        return y

    def get_context(self):
        return self.context[:, :, 0, 0]

    def forward(self, x):
        skip = x
        x = self.GlobalPooling(x)
        x = self.L2_norm(x)
        self.context = x
        x = self.upsample(x)
        skip = self.L2_norm(x)
        y = torch.cat((skip, x), 1)
        return y


class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, dilation=4, padding=4)
        self.conv3 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, dilation=8, padding=8)
        self.conv4 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, dilation=12, padding=12)
        self.relu = nn.ReLU(inplace=True)
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        fuse = torch.cat((conv1, conv2, conv3, conv4), 1)
        fuse = self.relu(fuse)
        y = self.ps(fuse)
        return y


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upscale_factor=2, padding=0):
        super(DeConv, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=upscale_factor, padding=padding)

    def forward(self, x):
        x = self.upsample(x)
        return x[:, :, 1:, 1:]


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, upscale_factor=2):
        super(UpConv, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class ConvGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(ConvGroup, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
