##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C

def merge_images(input, depth_target, depth_pred, d_min, d_max, orientation='row'):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    # d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    # d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    if orientation == 'row':
        img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    else:
        img_merge = np.vstack([rgb, depth_target_col, depth_pred_col])
    return img_merge


def imshow(img, height, width, mode='image'):
    _, _, h, w = img.shape
    img = torchvision.utils.make_grid(img, nrow=width)
    npimg = img.cpu().numpy() if img.is_cuda else img.numpy()
    fig = plt.figure(figsize=(w // 80 * width, h // 50 * height))
    if mode == 'image':
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    elif mode == 'depth':
        plt.imshow(npimg[0], cmap='plasma')
    else:
        plt.imshow(npimg[0], cmap='hot')
    return fig

def display_figure(params, writer, net, images, labels, depths, epoch):
    m = min(images.shape[0], 4)
    images = images[:m]
    labels = labels[:m]
    depths = depths[:m]
    # display image figure
    b, c, h, w = images.shape
    images = F.interpolate(images, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
    fig1 = imshow(images, height=1, width=b, mode='image')
    del images
    writer.add_figure('figure1-images', fig1, epoch)
    # display gt and pred figure
    labels = F.interpolate(labels, size=(h // 2, w // 2), mode='nearest')
    depths = F.interpolate(depths, size=(h // 2, w // 2), mode='nearest')
    fuse = torch.cat((labels, depths), 0)
    fig2 = imshow(fuse, height=2, width=b, mode='depth')
    del fuse, labels, depths
    writer.add_figure('figure2-depths', fig2, epoch)
    # display similarity figure
    if params.decoder in ['attention']:
        sim_map = net.decoder.get_sim_map().cpu()
        N = sim_map.shape[1]
        points = [N // 4, N // 2, 3 * N // 4]
        sim_pixels = sim_map[:, points]
        sim_pixels = sim_pixels.reshape((b, len(points), h//8, w//8))
        sim_pixels = sim_pixels.permute((1, 0, 2, 3))
        sim_pixels = sim_pixels.reshape((-1, 1, h//8, w//8))
        sim_pixels = F.interpolate(sim_pixels, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        fig3 = imshow(sim_pixels, height=2, width=b, mode='sim_map')
        writer.add_figure('figure3-pixel_attentions', fig3, epoch)
        del sim_map, sim_pixels
    return