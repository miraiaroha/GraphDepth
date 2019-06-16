##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: chenyuru
## This source code is licensed under the MIT-style license
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def create_lossfunc(args, net):
    ignore = None
    if args.dataset == 'kitti':
        ignore = 0
    criterion = net.LossFunc(args.min_depth, args.max_depth, args.classes, args.classifier, ignore_index=ignore)
    return criterion