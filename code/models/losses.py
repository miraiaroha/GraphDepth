import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, Q, P):
        """
            Args:
                P: ground truth probability distribution [batch, n, n]
                Q: predicted probability distribution [batch, n, n]
        """
        kl_loss = P * self.safe_log(P / Q)
        pixel_loss = torch.sum(kl_loss, dim=-1)
        total_loss = torch.mean(pixel_loss)
        return total_loss


class OrdinalRegression2d(nn.Module):
    def __init__(self, num_classes, ignore_index=0):
        super(OrdinalRegression2d, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        # self.target_index = list(np.arange(num_classes))
        # if ignore_index is not None:
        #     self.target_index.remove(ignore_index)

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, pred, label):
        """
            Args:
                pred: [batch, num_classes, h, w]
                label: [batch, h, w]
        """
        if self.ignore_index != None:
            mask = (label != self.ignore_index).float()
        else:
            mask = torch.ones_like(label, dtype=torch.float)
        label = label.unsqueeze(3).long()
        pred = pred.permute(0, 2, 3, 1)
        mask10 = (torch.arange(self.num_classes)).cuda() < label
        mask01 = (torch.arange(self.num_classes)).cuda() >= label
        mask10 = mask10.float()
        mask01 = mask01.float()
        entropy = self.safe_log(pred) * mask10 + \
            self.safe_log(1 - pred) * mask01
        pixel_loss = -torch.sum(entropy, -1)
        masked_pixel_loss = pixel_loss * mask
        total_loss = torch.sum(masked_pixel_loss) / mask.sum()
        return total_loss


class CrossEntropy2d(nn.Module):
    def __init__(self, num_classes, ignore_index=None):
        super(CrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def safe_log(self, x):
        return torch.log(torch.clamp(x, 1e-8, 1e8))

    def forward(self, pred, label):
        """
            Args:
                pred: [batch, num_classes, h, w]
                label: [batch, h, w]
        """
        if self.ignore_index != None:
            mask = (label != self.ignore_index).float()
        else:
            mask = torch.ones_like(label, dtype=torch.float)
        label = label.unsqueeze(3).long()
        pred = F.softmax(pred, 1).permute(0, 2, 3, 1)
        one_hot_label = (torch.arange(self.num_classes)).cuda() == label
        one_hot_label = one_hot_label.float()
        entropy = one_hot_label * self.safe_log(pred) + \
            (1 - one_hot_label) * self.safe_log(1 - pred)
        pixel_loss = - torch.sum(entropy, -1)
        masked_pixel_loss = pixel_loss * mask
        total_loss = torch.sum(masked_pixel_loss) / mask.sum()
        return total_loss


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=0, thresh=0.5, min_kept=0):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predict, target):
        """
            Args:
                predict:(batch, num_classes, h, w)
                target:(batch, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(
            0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(
            1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(
            2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_index
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Max prob: {:.4f}, hard ratio: {} = {} / {} '.format(input_prob.max(), round(len(valid_inds) /
                                                                                               num_valid, 4), len(valid_inds), num_valid))

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_index)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_index
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(
            input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)
