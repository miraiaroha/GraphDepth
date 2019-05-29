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

def create_data_loaders(args):
    # Data loading code
    print("=> creating data loaders ...")
    nyu_path = '/ssd/datasets/nyu-depth-v2/nyudepthv2'
    traindir = os.path.join(nyu_path, 'train')
    valdir = os.path.join(nyu_path, 'val')

    #traindir = os.path.join('data', args.data, 'train')
    #valdir = os.path.join('data', args.data, 'val')
    train_loader = None
    val_loader = None

    if args.dataset == 'nyu':
        from dataloaders.nyu_dataloader import NYUDataset
        if args.mode == 'train':
            train_dataset = NYUDataset(args.trainp, args.valp, args.traint, args.valt, args.min_depth, args.max_depth, mode='train')
        val_dataset = NYUDataset(valdir, type='val', modality=args.modality, sparsifier=sparsifier)

    elif args.data == 'kitti':
        from dataloaders.kitti_dataloader import KITTIDataset
        if not args.evaluate:
            train_dataset = KITTIDataset(traindir, type='train',
                modality=args.modality, sparsifier=sparsifier)
        val_dataset = KITTIDataset(valdir, type='val',
            modality=args.modality, sparsifier=sparsifier)

    else:
        raise RuntimeError('Dataset not found.' +
                           'The dataset must be either of nyudepthv2 or kitti.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    # put construction of train loader here, for those who are interested in testing only
    if not args.evaluate:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None,
            worker_init_fn=lambda work_id:np.random.seed(work_id))
            # worker_init_fn ensures different sampling patterns for each data loading thread

    print("=> data loaders created.")
    return train_loader, val_loader

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