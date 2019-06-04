import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from adabound import AdaBound
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

    if args.mode == 'train':
        train_dataset = TargetDataset(args.rgbdir, args.depdir, args.trainrgb, args.traindep, args.min_depth, args.max_depth, mode=args.mode)
    if args.mode in ['train', 'val']:
        val_dataset = TargetDataset(args.rgbdir, args.depdir, args.valrgb, args.valdep, args.min_depth, args.max_depth, mode=args.mode)
    if args.mode == 'test':
        test_dataset = TargetDataset(args.rgbdir, args.depdir, args.testrgb, args.testdep, args.min_depth, args.max_depth, mode=args.mode)
    
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
    if args.optim == 'sgd':
        return optim.SGD(optim_params, args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        return optim.Adagrad(optim_params, args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    elif args.optim == 'amsgrad':
        return optim.Adam(optim_params, args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay, amsgrad=True)
    elif args.optim == 'adabound':
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma,
                        weight_decay=args.weight_decay)
    else:
        assert args.optim == 'amsbound'
        return AdaBound(optim_params, args.lr, betas=(args.beta1, args.beta2),
                        final_lr=args.final_lr, gamma=args.gamma, 
                        weight_decay=args.weight_decay, amsbound=True)

def create_scheduler(args, optimizer):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[12], gamma=0.1)
    elif args.scheduler == 'poly':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lrd)
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
                                     )
    if args.mode == 'finetune':
        if args.encoder == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif args.encoder == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        Trainer.reload(resume=resnet, mode='finetune')
    if args.mode == 'retain':
        Trainer.reload(resume=args.resume, mode='retain')
    
    Trainer.train()
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
    datastes = create_datasets(args)
    # Network
    net = create_network(args)
    # Loss Function
    criterion = create_lossfunc(args, net)
    # optimizer parameters
    optim_params = create_params(args, net)
    optimizer = create_optimizer(args, optim_params)
    # learning rate scheduler
    scheduler = create_scheduler(args, optimizer)

    if args.mode == 'test':
        test(args, net, datastes, criterion)
    else:
        train(args, net, datastes, criterion, optimizer, scheduler)

    
if __name__ == '__main__':
    main()