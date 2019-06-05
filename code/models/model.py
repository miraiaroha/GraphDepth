import os
import sys
sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from gcdecoder import GCDecoder
from sadecoder import SADecoder
from losses import OrdinalRegression2d, CrossEntropy2d, OhemCrossEntropy2d

def continuous2discrete(continuous_depth, Min, Max, num_classes):
    if continuous_depth.is_cuda:
        Min, Max, NC = Min.cuda(), Max.cuda(), num_classes.cuda()
    continuous_depth = torch.clamp(continuous_depth, Min.item(), Max.item())
    discrete_depth = torch.round(torch.log(continuous_depth / Min) / torch.log(Max / Min) * (NC - 1))
    return discrete_depth

def discrete2continuous(discrete_depth, Min, Max, num_classes):
    if discrete_depth.is_cuda:
        Min, Max, NC = Min.cuda(), Max.cuda(), num_classes.cuda()
    continuous_depth = torch.exp(discrete_depth / (NC - 1) * torch.log(Max / Min) + torch.log(Min))
    return continuous_depth

class ClassificationModel(nn.Module):
    def __init__(self, min_depth, max_depth, num_classes, classifierType, inferenceType):
        super().__init__()
        self.min_depth = torch.tensor(min_depth, dtype=torch.float)
        self.max_depth = torch.tensor(max_depth, dtype=torch.float)
        self.num_classes = torch.tensor(num_classes, dtype=torch.float)
        self.classifierType = classifierType
        self.inferenceType = inferenceType

    def decode_ord(self, y):
        batch_size, prob, height, width = y.shape
        y = torch.reshape(y, (batch_size, self.num_classes, 2, height, width))
        denominator = torch.sum(torch.exp(y), 2)
        pred_score = torch.div(torch.exp(y[:, :, 1, :, :]), denominator)
        return pred_score

    def hard_cross_entropy(self, pred_score, Min, Max, num_classes):
        pred_label = torch.argmax(pred_score, 1, keepdim=True).float()
        pred_depth = discrete2continuous(pred_label, Min, Max, num_classes)
        return pred_depth

    def soft_cross_entropy(self, pred_score, Min, Max, num_classes):
        if pred_score.is_cuda:
            Min, Max = Min.cuda(), Max.cuda()
        pred_prob = F.softmax(pred_score, dim=1)
        nc = torch.arange(num_classes.item())
        if pred_score.is_cuda:
            nc = nc.cuda()
        weight = nc * torch.log(Max / Min) / (NC - 1) + torch.log(Min)
        weight = weight.unsqueeze(-1)
        pred_prob = pred_prob.permute((0, 2, 3, 1))
        output = torch.exp(torch.matmul(pred_prob, weight))
        output = output.permute((0, 3, 1, 2))
        return output

    def hard_ordinal_regression(self, pred_prob, Min, Max, num_classes):
        mask = (pred_prob > 0.5).float()
        pred_label = torch.sum(mask, 1, keepdim=True)
        #pred_label = torch.round(torch.sum(pred_prob, 1, keepdim=True))
        pred_depth = (discrete2continuous(pred_label, Min, Max, num_classes) +
                      discrete2continuous(pred_label + 1, Min, Max, num_classes)) / 2
        return pred_depth

    def soft_ordinal_regression(self, pred_prob, Min, Max, num_classes):
        pred_prob_sum = torch.sum(pred_prob, 1, keepdim=True)
        Intergral = torch.floor(pred_prob_sum)
        Fraction = pred_prob_sum - Intergral
        depth_low = (discrete2continuous(Intergral, Min, Max, num_classes) +
                     discrete2continuous(Intergral + 1, Min, Max, num_classes)) / 2
        depth_high = (discrete2continuous(Intergral + 1, Min, Max, num_classes) +
                      discrete2continuous(Intergral + 2, Min, Max, num_classes)) / 2
        pred_depth = depth_low * (1 - Fraction) + depth_high * Fraction
        return pred_depth

    def inference(self, y):
        # mode
        # OR = Ordinal Regression
        # CE = Cross Entropy
        if self.classifierType == 'OR':
            if self.inferenceType == 'soft':
                inferenceFunc = self.soft_ordinal_regression
            else:    # hard OR
                inferenceFunc = self.hard_ordinal_regression
        else:  # 'CE'
            if self.inferenceType == 'soft': # soft CE
                inferenceFunc = self.soft_cross_entropy
            else:     # hard CE
                inferenceFunc = self.hard_cross_entropy
        pred_depth = inferenceFunc(y, self.min_depth, self.max_depth, self.num_classes)
        return pred_depth

    def forward():
        raise NotImplementedError

    
def make_decoder(decoder='graph 2048 128 1'):
    command = decoder.split(' ')
    if command[0] == 'graph':
        dec = GCDecoder
    elif command[0] == 'attention':
        dec = SADecoder
    else:
        raise RuntimeError('decoder not found.' +
                           'The decoder must be either graph or attention.')
    return dec(*[int(x) for x in command[1:]])

def make_classifier(classifierType='OR', num_classes=80, in_channel=2048):
    if classifierType == 'CE':
        channel = num_classes 
    elif classifierType == 'OR':
        channel = 2 * num_classes
    else:
        raise RuntimeError('classifier not found.' +
                           'The classifier must be either of CE or OR.')
    classifier = nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channel, channel, kernel_size=1, stride=1)),
        ('upsample', nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)),
        ]))
    return classifier


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

class ResNet(ClassificationModel):
    def __init__(self, min_depth, max_depth, num_classes,
                 classifierType, inferenceType, decoderType,
                 layers=[3, 4, 6, 3], 
                 block=Bottleneck):
        # Note: classifierType: CE=Cross Entropy, OR=Ordinal Regression
        super(ResNet, self).__init__(min_depth, max_depth, num_classes, classifierType, inferenceType)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # fixed the parameters before
        # for p in self.parameters():
        #     p.requires_grad = False
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False
        # extra added layers
        self.decoder = make_decoder(decoderType)
        self.classifier = make_classifier(classifierType, num_classes, 2048)
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

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.decoder(x)
        y = self.classifier(x)
        if self.classifierType == 'OR':
            y = self.decode_ord(y)
        return y

    class LossFunc(nn.Module):
        def __init__(self, min_depth, max_depth, num_classes, classifierType, ohem=False, ignore_index=None):
            super(ResNet.LossFunc, self).__init__()
            self.min_depth = torch.tensor(min_depth, dtype=torch.float)
            self.max_depth = torch.tensor(max_depth, dtype=torch.float)
            self.num_classes = torch.tensor(num_classes, dtype=torch.float)
            self.ignore_index = ignore_index
            if classifierType == 'OR': # 'Ordinal Regression
                self.AppearanceLoss = OrdinalRegression2d(ignore_index=ignore_index)
            elif classifierType == 'CE':  # 'Cross Entropy'
                if ignore_index is None:
                    ignore_index = num_classes + 1
                self.AppearanceLoss = CrossEntropy2d(ignore_index)
            elif classifierType == 'OHEM': # 'Online Hard Example Mining'
                if ignore_index is None:
                    ignore_index = num_classes + 1
                    self.AppearanceLoss = OhemCrossEntropy2d(ignore_index) 
            else:
                raise RuntimeError('classifier not found.' +
                           'The classifier must be either of OR, CE or OHEM.')

        def forward(self, pred_score, label, sim_map, epoch):
            """
                Args:
                    pred: [batch, num_classes, h, w]
                    label: [batch, 1, h, w]
            """
            label = continuous2discrete(label, self.min_depth, self.max_depth, self.num_classes)
            # image loss
            image_loss = self.AppearanceLoss(pred_score, label.squeeze(1).long())
            return image_loss
    
