import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from graphdepth_utils import load_params_from_parser

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



def main():
    args = load_params_from_parser()

    # Dataset
    trainset = HyperECUST_FI(params['trainset_path'],
                             params['trainset_txt'])
    validset = HyperECUST_FV(params['validset_path'],
                             params['validset_txt'], snr=1 - s)
    testset = HyperECUST_FV(params['testset_path'],
                            params['testset_txt'], snr=1 - s)
    datasets = {'train': trainset, 'valid': validset, 'test': testset}
    # Define model
    net = MobileFacenet(trainset.class_nums)
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
    # Define criterion
    criterion = net.LossFunc()
    # Define trainer
    Trainer = MobileFacenetTrainer(params=params,
                                   net=net,
                                   datasets=datasets,
                                   optimizer=optimizer,
                                   lr_scheduler=exp_lr_scheduler,
                                   criterion=criterion,
                                   workspace_dir=params['workspace_dir'],
                                   log_dir=params['log_dir'],
                                   sets=list(datasets.keys()),
                                   eval_func=Evaluation_10_fold,
                                   )

    # Train model
    if args.mode == 'train':
        Trainer.train()
        return

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