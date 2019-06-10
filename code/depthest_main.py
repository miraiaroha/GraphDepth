import os
import sys
import argparse
import json
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from adabound import AdaBound
import torchvision
import torchvision.models as models
from torch.optim import lr_scheduler

from depthest_utils import load_params_from_parser, compute_errors
from trainers.depthest_trainer import DepthEstimationTrainer
from models.model import ResNet

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def create_datasets(args):
    # Create dataset code
    print("=> creating datasets ...")
    train_dataset = None
    val_dataset = None
    test_dataset = None

    if args.dataset == 'nyu':
        from dataloaders.nyu_dataloader import NYUDataset as TargetDataset
    elif args.dataset == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset as TargetDataset
    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyu or kitti.')
    if args.mode == 'test':
        test_dataset = TargetDataset(args.rgbdir, args.depdir, args.testrgb, args.testdep, args.min_depth, args.max_depth, mode='test')
    else:
        train_dataset = TargetDataset(args.rgbdir, args.depdir, args.trainrgb, args.traindep, args.min_depth, args.max_depth, mode='train')
        val_dataset = TargetDataset(args.rgbdir, args.depdir, args.valrgb, args.valdep, args.min_depth, args.max_depth, mode='val')
    
    print("<= datasets created.")
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    return datasets

def create_network(args):
    resnet_layer_settings = {'50':  [3, 4, 6, 3], 
                             '101': [3, 4, 23, 3]}
    if args.encoder == 'resnet50':
        setttings = resnet_layer_settings['50']
    elif args.encoder == 'resnet101':
        setttings = resnet_layer_settings['101']
    else:
        raise RuntimeError('network not found.' +
                           'The network must be either of resnet50 or resnet101.')

    net = ResNet(args.min_depth, args.max_depth, args.classes, args.classifier, args.inference, args.decoder, setttings)
    return net

def create_lossfunc(args, net):
    ignore = None
    if args.dataset == 'kitti':
        ignore = 0
    criterion = net.LossFunc(args.min_depth, args.max_depth, args.classes, args.classifier, ignore_index=ignore)
    return criterion

def create_params(args, net):
    def get_params(mod, Type=''):
        params = []
        for m in mod:
            for n, p in m.named_parameters():
                if Type in n:
                    params += [p]
        return params
    # filter manully
    if args.encoder in ['resnet50', 'resnet101']:
        base_modules = list(net.children())[:8]
        base_params = get_params(base_modules, '')
        base_params = filter(lambda p: p.requires_grad, base_params)
        add_modules = list(net.children())[8:]
        add_weight_params = get_params(add_modules, 'weight')
        add_bias_params = get_params(add_modules, 'bias')
        if args.optimizer in ['adabound', 'amsbound']:
            optim_params = [{'params': base_params, 'lr': args.lr, 'final_lr': args.flr},
                            {'params': add_weight_params, 'lr': args.lr*10, 'final_lr': args.flr},
                            {'params': add_bias_params, 'lr': args.lr*20, 'final_lr': args.flr*10}]
        else:
            optim_params = [{'params': base_params, 'lr': args.lr},
                            {'params': add_weight_params, 'lr': args.lr*10},
                            {'params': add_bias_params, 'lr': args.lr*20}]
    return optim_params

def create_optimizer(args, optim_params):
    if args.optimizer == 'sgd':
        return optim.SGD(optim_params, args.lr, momentum=args.momentum,
                         weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        return optim.Adagrad(optim_params, args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wd)
    elif args.optimizer == 'amsgrad':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.wd, amsgrad=True)
    elif args.optimizer == 'adabound':
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.flr, gamma=args.gamma,
                        weight_decay=args.wd)
    else:
        assert args.optimizer == 'amsbound'
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma, 
                        weight_decay=args.wd, amsbound=True)


def create_scheduler(args, optimizer, datasets):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[28, 45], gamma=0.1)
    elif args.scheduler == 'poly':
        total_step = len(datasets['train']) / args.batch * args.epochs
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: (1-x/total_step) ** 0.9)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    return scheduler
     
def train(args, net, datasets, criterion, optimizer, scheduler):
    # Define trainer
    Trainer = DepthEstimationTrainer(params=args,
                                     net=net,
                                     datasets=datasets,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     sets=list(datasets.keys()),
                                     eval_func=compute_errors,
                                     disp_func=display_figure
                                     )
    if args.mode == 'finetune':
        if args.encoder == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif args.encoder == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        Trainer.reload(resume=resnet, mode='finetune')
    elif args.mode == 'retain':
        Trainer.reload(resume=args.resume, mode='retain')

    Trainer.train()
    return

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

def test(args, net, datasets, criterion):
    # Define trainer
    Trainer = DepthEstimationTrainer(params=args,
                                     net=net,
                                     datasets=datasets,
                                     criterion=None,
                                     optimizer=None,
                                     scheduler=None,
                                     sets=list(datasets.keys()),
                                     eval_func=compute_errors,
                                     )
    Trainer.reload(resume=args.resume, mode='test')
    Trainer.test()
    return

def main():
    args = load_params_from_parser()
    # Dataset
    datasets = create_datasets(args)
    # Network
    net = create_network(args)
    # Loss Function
    criterion = create_lossfunc(args, net)
    # optimizer parameters
    optim_params = create_params(args, net)
    optimizer = create_optimizer(args, optim_params)
    # learning rate scheduler
    scheduler = create_scheduler(args, optimizer, datasets)
    if args.mode == 'test':
        test(args, net, datasets, criterion)
    else: # train, retain, finetune
        train(args, net, datasets, criterion, optimizer, scheduler)

    
if __name__ == '__main__':
    main()