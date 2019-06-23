##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import pi, sqrt

safe_log = lambda x: torch.log(torch.clamp(x, 1e-8, 1e8))

class BaseClassifier2d_(nn.Module):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        """
        Parameter
        ---------
        ignore_index : Specifies a target value that is ignored
                       and does not contribute to the input gradient
        reduction : Specifies the reduction to apply to the output: 
                    'mean' | 'sum'. 'mean': elemenwise mean, 
                    'sum': class dim will be summed and batch dim will be averaged.
        use_weight : whether to use weights of classes.
        weight : Tensor, optional
                a manual rescaling weight given to each class.
                If given, has to be a Tensor of size "nclasses"
        """
        super(BaseClassifier2d_, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_weights = use_weights
        if use_weights:
            print("w/ class balance")
            print(weight)
            self.weight = torch.FloatTensor(weight).cuda()
        else:
            #print("w/o class balance")
            self.weight = None

    def get_entropy(self, pred, label):
        raise NotImplementedError

    def forward(self, pred, label):
        """
        Parameter
        ---------
        pred: [batch, num_classes, h, w]
        label: [batch, h, w]
        """
        assert not label.requires_grad
        assert pred.dim() == 4
        assert label.dim() == 3
        assert pred.size(0) == label.size(0), "{0} vs {1} ".format(pred.size(0), label.size(0))
        assert pred.size(2) == label.size(1), "{0} vs {1} ".format(pred.size(2), label.size(1))
        assert pred.size(3) == label.size(2), "{0} vs {1} ".format(pred.size(3), label.size(3))

        n, c, h, w = pred.size()
        if self.use_weights and self.weight is None:
            print('label size {}'.format(label.shape))
            freq = np.zeros(c)
            for k in range(c):
                mask = (label[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq) * c
            weight = np.median(weight) / weight
            self.weight = torch.FloatTensor(weight).cuda()
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = 1

        entropy = self.get_entropy(pred, label)

        if self.ignore_index != None:
            mask = (label != self.ignore_index).float()
        else:
            mask = torch.ones_like(label, dtype=torch.float)

        weighted_entropy = entropy * self.weight
        if self.reduction == 'sum':
            pixel_loss = torch.sum(weighted_entropy, -1) * mask
            loss = torch.sum(pixel_loss) / mask.sum()
        elif self.reduction == 'mean':
            pixel_loss = torch.mean(weighted_entropy, -1) * mask
            loss = torch.sum(pixel_loss) / mask.sum()
        return loss


class OrdinalRegression2d(BaseClassifier2d_):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None):
        super(OrdinalRegression2d, self).__init__(ignore_index, reduction, use_weights, weight)

    def get_entropy(self, pred, label):
        n, c, h, w = pred.size()
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        mask10 = ((torch.arange(c)).cuda() <  label).float()
        mask01 = ((torch.arange(c)).cuda() >= label).float()
        entropy = safe_log(pred) * mask10 + safe_log(1 - pred) * mask01
        return -entropy

def NormalDist(x, sigma):
    f = torch.exp(-x**2/(2*sigma**2)) / sqrt(2*pi*sigma**2)
    return f

class CrossEntropy2d(BaseClassifier2d_):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None,
                 eps=0.0, priorType='uniform'):
        """
        Parameter
        ---------
        eps : label smoothing factor
        prior : prior distribution, if uniform equivalent to the 
                label smoothing trick (https://arxiv.org/abs/1512.00567).
                However, gaussian distribution is more friendly with the depth estimation.
        """
        super(CrossEntropy2d, self).__init__(ignore_index, reduction, use_weights, weight)
        self.eps = eps
        self.priorType = priorType

    def get_entropy(self, pred, label):
        n, c, h, w = pred.size()
        label = label.unsqueeze(3).long()
        pred = F.softmax(pred, 1).permute(0, 2, 3, 1)
        one_hot_label = ((torch.arange(c)).cuda() == label).float()

        if self.eps == 0:
            prior = 0
        else:
            if self.priorType == 'gaussian':
                tensor = (torch.arange(c).cuda() - label).float()
                prior = NormalDist(tensor, c / 10)
            elif self.priorType == 'uniform':
                prior = 1 / (c-1)

        smoothed_label = (1 - self.eps) * one_hot_label + self.eps * prior * (1 - one_hot_label)
        entropy = smoothed_label * safe_log(pred) + (1 - smoothed_label) * safe_log(1 - pred)
        return -entropy 

class OhemCrossEntropy2d(CrossEntropy2d):
    def __init__(self, ignore_index=None, reduction='sum', use_weights=False, weight=None,
                 eps=0.0, priorType='uniform', thresh=0.6, min_kept=0, 
                 ):
        """
        Parameter
        ---------
        thresh : OHEM (online hard example mining) threshold of correct probability
        min_kept : OHEM of minimal kept pixels

        Description
        -----------
        modified from https://github.com/PkuRainBow/OCNet.pytorch/blob/master/utils/loss.py#L68
        """
        super(OhemCrossEntropy2d, self).__init__(ignore_index, reduction, use_weights, weight,
                                                 eps, priorType)
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)

    def get_ohem_label(self, pred, label):
        n, c, h, w = pred.size()
        if self.ignore_index is None:
            self.ignore_index = c + 1

        input_label = label.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(pred.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0, keepdims=True))
        input_prob /= input_prob.sum(axis=0, keepdims=True)

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        valid_label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            valid_prob = input_prob[:,valid_flag]
            valid_prob = valid_prob[valid_label, np.arange(len(valid_label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = valid_prob.argsort()
                threshold_index = index[ min(len(index), self.min_kept) - 1 ]
                if valid_prob[threshold_index] > self.thresh:
                    threshold = valid_prob[threshold_index]
            kept_flag = valid_prob <= threshold
            valid_kept_inds = valid_inds[kept_flag]
            valid_inds = valid_kept_inds

        self.ohem_ratio = len(valid_inds) / num_valid
        #print('Max prob: {:.4f}, hard ratio: {:.4f} = {} / {} '.format(input_prob.max(), self.ohem_ratio, len(valid_inds), num_valid))
        valid_kept_label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = valid_kept_label
        #valid_flag_new = input_label != self.ignore_index
        # print(np.sum(valid_flag_new))
        label = torch.from_numpy(input_label.reshape(label.size())).long().cuda()
        return label
        
    def get_entropy(self, pred, label):
        label = self.get_ohem_label(pred, label)
        entropy = super(OhemCrossEntropy2d, self).get_entropy(pred, label)
        return entropy
