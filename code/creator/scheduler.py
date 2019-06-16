##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from torch.optim import lr_scheduler

def create_scheduler(args, optimizer, datasets):
    if args.scheduler == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[28, 45], gamma=args.lr_decay)
    elif args.scheduler == 'poly':
        total_step = len(datasets['train']) / args.batch * args.epochs
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda x: (1-x/total_step) ** args.poly)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    return scheduler