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
from model import ResNet

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
    
    print("=> datasets created.")
    datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    return datasets

def create_network(args):
    resnet_layer_settings = {'50':  [3, 4, 6, 3], 
                             '101': [3, 4, 23, 3]}
    if args.mode == 'resnet50':
        setttings = resnet_layer_settings['50']
    elif args.mode == 'resnet101':
        setttings = resnet_layer_settings['101']
    else:
        raise RuntimeError('network not found.' +
                           'The network must be either of resnet50 or resnet101.')

    net = ResNet(args.min_depth, args.max_depth, args.num_classes, 
                    args.classifier, args.inference, args.decoder, setttings)
    return net

def create_lossfunc(args, net):
    ignore = None
    if args.dataset == 'kitti':
        ignore = 0
    criterion = net.LossFunc(args.min_depth, args.max_depth, args.num_classes, args.classifier, ignore_index=ignore)
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
        add_modules = list(myModel.children())[8:]
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
    
def train(args, net, datasets, criterion, optimizer, scheduler, isFineTune=False):
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
    if isFineTune:
        if args.encoder == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif args.encoder == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        Trainer.reload(resume=resnet, finetune=True)
    
    Trainer.train()
    return

def test(args, net, datasets):
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
    Trainer.load_checkpoint(params['resume'])
    Trainer.test(0.4)
    return

    if isFineTune:
        
        # Finetune
        # Load the pretrained model
        # @ Note: the optimizer and lr_scheduler should be redefined if execute Trainer.reload(True),
        
        net = Trainer.net
        # Define optimizers
        ignored_params = list(map(id, net.linear1.parameters()))
        ignored_params += [id(net.weight)]
        prelu_params = []
        for m in net.modules():
            if isinstance(m, nn.PReLU):
                ignored_params += list(map(id, m.parameters()))
                prelu_params += m.parameters()
        base_params = filter(lambda p: id(p) not in ignored_params,
                            net.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'weight_decay': 4e-5},
            {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
            {'params': net.weight, 'weight_decay': 4e-4},
            {'params': prelu_params, 'weight_decay': 0.0}
        ], lr=0.01, momentum=0.9, nesterov=True)
        exp_lr_scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[12], gamma=0.1)

        Trainer.optimizer = optimizer
        Trainer.lr_scheduler = exp_lr_scheduler
        Trainer.train()
        del Trainer
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
    if args.mode == 'train':
        train(args, net, datastes, criterion, optimizer, scheduler, False)
    elif args.mode == 'test':
        test(args, net, datastes, criterion)
    elif args.mode == 'finetune':
        train(args, net, datastes, criterion, optimizer, scheduler, True)

    

    # Evaluate one epoch
    # if args.mode == 'eval':
    #     Trainer.reload()
    #     acc = Trainer.eval_epoch()
    #     print("acc {:.4f}".format(acc))
    #     return

    # Test model
    if args.mode == 'test':
        Trainer.load_checkpoint(params['resume'])
        Trainer.test(0.4)
        return

    # Finetune
    # Load the pretrained model
    # @ Note: the optimizer and lr_scheduler should be redefined if execute Trainer.reload(True),
    Trainer.reload(finetune=True)
    net = Trainer.net
    # Define optimizers
    ignored_params = list(map(id, net.linear1.parameters()))
    ignored_params += [id(net.weight)]
    prelu_params = []
    for m in net.modules():
        if isinstance(m, nn.PReLU):
            ignored_params += list(map(id, m.parameters()))
            prelu_params += m.parameters()
    base_params = filter(lambda p: id(p) not in ignored_params,
                         net.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'weight_decay': 4e-5},
        {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
        {'params': net.weight, 'weight_decay': 4e-4},
        {'params': prelu_params, 'weight_decay': 0.0}
    ], lr=0.01, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[12], gamma=0.1)

    Trainer.optimizer = optimizer
    Trainer.lr_scheduler = exp_lr_scheduler
    Trainer.train()
    del Trainer
    return
    

if __name__ == '__main__':
    main()