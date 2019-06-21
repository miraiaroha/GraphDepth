##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from math import ceil

def compute_errors(gt, pred):
    """
    Parameters
    ----------
    gt : torch.Tensor
        shape [batch, h, w]
    pred : torch.Tensor
        shape [batch, h, w]

    Return
    ------
    measures : dict
    """
    safe_log = lambda x: torch.clamp(x, 1e-6, 1e6)
    safe_log10 = lambda x: torch.clamp(x, 1e-6, 1e6)
    batch_size = pred.shape[0]
    mask = gt > 0
    gt = gt[mask]
    pred = pred[mask]
    thresh = torch.max(gt / pred, pred / gt)
    a1 = (thresh < 1.25).float().mean() * batch_size
    a2 = (thresh < 1.25 ** 2).float().mean() * batch_size
    a3 = (thresh < 1.25 ** 3).float().mean() * batch_size

    rmse = ((gt - pred) ** 2).mean().sqrt() * batch_size
    rmse_log = ((safe_log(gt) - safe_log(pred))** 2).mean().sqrt() * batch_size
    log10 = (safe_log10(gt) - safe_log10(pred)).abs().mean() * batch_size
    abs_rel = ((gt - pred).abs() / gt).mean() * batch_size
    sq_rel = ((gt - pred)**2 / gt).mean() * batch_size
    measures = {'a1': a1, 'a2': a2, 'a3': a3, 'rmse': rmse,
                'rmse_log': rmse_log, 'log10': log10, 'abs_rel': abs_rel, 'sq_rel': sq_rel}
    return measures


def pad_image(image, target_size):
    """
    Parameters
    ----------
    image : numpy.ndarray 
          shape [batch_size, c, h, w]
    target_size : tuple or list

    Description
    -----------
    Pad an image up to the target size.
    """
    rows_missing = target_size[0] - image.shape[2]
    cols_missing = target_size[1] - image.shape[3]
    padded_img = np.pad(image, ((0, 0), (0, 0), (0, rows_missing), (0, cols_missing)), 'constant')
    return padded_img

def predict_sliding_(net, image, tile_size, classes, scale=1):
    """
    Parameters
    ----------
    net : nn.Module
    image : torch.Tensor
            shape [batch_size, c, h, w]
    tile_size: tuple or list
            max size of image inputted to the net
    scale : scalar

    Return
    ------
    full_probs : numpy.ndarray
                 shape [batch_size, classes, h, w]
    Description
    -----------
    Predict the whole image using multiple crops.
    The scale specify whether rescale the input image before predicting the results.
    """
    image = image.cpu().numpy()
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image
    N_, C_, H_, W_ = scaled_img.shape
    full_probs = np.zeros((N_, classes, H_, W_))
    count_predictions = np.zeros((N_, classes, H_, W_))
    overlap = 0
    stride_h = ceil(tile_size[0] * (1 - overlap))
    stride_w = ceil(tile_size[1] * (1 - overlap))
    tile_rows = int(ceil((H_ - tile_size[0]) / stride_h) + 1)  # strided convolution formula
    tile_cols = int(ceil((W_ - tile_size[1]) / stride_w) + 1)
    #print("Need {} x {} prediction tiles @ stride {} px, {} py".format(tile_rows, tile_cols, stride_h, stride_w))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride_w)
            y1 = int(row * stride_h)
            x2 = min(x1 + tile_size[1], W_)
            y2 = min(y1 + tile_size[0], H_)
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows

            img = scaled_img[:, :, y1:y2, x1:x2]
            padded_img = pad_image(img, tile_size)
            tile_counter += 1
            #print("Predicting tile {}".format(tile_counter))
            padded_prediction_ = net(torch.from_numpy(padded_img).cuda())
            padded_prediction = padded_prediction_
            # padded_prediction = nn.functional.softmax(padded_prediction, dim=1)
            padded_prediction = F.upsample(padded_prediction, size=tile_size, mode='bilinear', align_corners=True)
            padded_prediction = padded_prediction.cpu().data.numpy()
            prediction = padded_prediction[:, :, :img.shape[2], :img.shape[3]]
            count_predictions[:, :, y1:y2, x1:x2] += 1
            full_probs[:, :, y1:y2, x1:x2] += prediction 
            torch.cuda.empty_cache()

    full_probs /= count_predictions
    full_probs = ndimage.zoom(full_probs, (1., 1., 1./scale, 1./scale), order=1, prefilter=False)
    return full_probs


def predict_whole_img_(net, image, scale):
    """
    Parameters
    ----------
    net : nn.Module
    image : torch.Tensor
            shape [batch_size, c, h, w]
    scale : scalar

    Return
    ------
    full_probs : numpy.ndarray
                 shape [batch_size, classes, h, w]      
    Description
    -----------
    Predict the whole image w/o using multiple crops.
    The scale specify whether rescale the input image before predicting the results.
    """
    image = image.cpu().numpy()
    _, _, H_, W_ = image.shape
    if scale != 1:
        scaled_img = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)
    else:
        scaled_img = image

    full_probs_ = net(torch.from_numpy(scaled_img).cuda())
    full_probs = full_probs_
    full_probs = F.upsample(full_probs, size=(H_, W_), mode='bilinear', align_corners=True)
    full_probs = full_probs.cpu().data.numpy()
    return full_probs

def predict_sliding(net, image, tile_size, classes, scale=1, flip=False):
    probs = predict_sliding_(net, image, tile_size, classes, scale=scale)
    if flip:
        flipped_probs = predict_sliding_(net, image.flip([3]), tile_size, classes, scale=scale)
        probs = 0.5 * (probs + flipped_probs[:,:,:,::-1])
    return probs

def predict_whole_img(net, image, scale=1, flip=False):
    probs = predict_whole_img_(net, image, scale=scale)
    if flip:
        flipped_probs = predict_whole_img_(net, image.flip([3]), scale=scale)
        probs = 0.5 * (probs + flipped_probs[:,:,:,::-1])
    return probs

def predict_multi_scale(net, image, scales, classes, flip):
    """
    Parameters
    ----------
    net : nn.Module
    image : numpy.ndarray 
          shape [batch_size, c, h, w]
    scales : list
    flip : bool

    Description
    -----------
    Predict an image by looking at it with different scales.
    We choose the "predict_whole_img" for the image with less than the original input size,
    for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    N_, C_, H_, W_ = image.shape
    full_probs = np.zeros((N_, classes, H_, W_))  
    for scale in scales:
        scale = float(scale)
        scaled_ = [int(x*scale) for x in [H_, W_]]
        #print("Predicting image scaled by {}, [h, w] = [{}, {}]".format(scale, *scaled_))
        sys.stdout.flush()
        if scale <= 1.0:
            scaled_probs = predict_whole_img(net, image, scale=scale, flip=flip)
        else:        
            scaled_probs = predict_sliding(net, image, (H_, W_), classes, scale=scale, flip=flip)
        full_probs += scaled_probs
    full_probs /= len(scales)
    return full_probs