import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

cmap = plt.cm.viridis

def load_params_from_parser():
    modes = ['train', 'test', 'finetune', 'retain']
    encoder_names = ['resnet50', 'resnet101']
    decoder_names = ['graph', 'attention']
    classifier_type = ['CE', 'OR', 'OHEM']
    inference_type = ['hard', 'soft']
    loss_names = ['l1', 'l2']
    opt_names = ['sgd', 'adam', 'adagrad', 'amsgrad', 'adabound', 'amsbound']
    sch_names = ['step', 'poly', 'plateau']
    dataset_names = ['nyu', 'kitti']

    parser = argparse.ArgumentParser(description='GraphDepth Pytorch Implementation.')
    parser.add_argument('--mode',                          type=str,   help='mode: '+'|'.join(modes)+' (default: fintune)', default='finetune',  choices=modes)
    parser.add_argument('--encoder',                       type=str,   help='encoder: '+'|'.join(encoder_names)+' (default: resnet50)', default='resnet50', choices=encoder_names)
    parser.add_argument('--decoder',                       type=str,   help='decoder: '+'|'.join(decoder_names)+' (default: graph)', default='graph', choices=decoder_names)
    parser.add_argument('--classifier',                    type=str,   help='classifier: '+'|'.join(classifier_type)+' (default: OR)', default='OR', choices=classifier_type)
    parser.add_argument('--inference',                     type=str,   help='inference: '+'|'.join(inference_type)+' (default: soft)', default='soft', choices=inference_type)
    # dataset
    parser.add_argument('--dataset',                       type=str,   help='dataset: '+'|'.join(dataset_names)+' (default: nyu)', default='nyu', choices=dataset_names)
    parser.add_argument('--rgbdir',                        type=str,   help='root to rgb', required=True)
    parser.add_argument('--depdir',                        type=str,   help='root to depth', required=True)
    parser.add_argument('--train-rgb',    dest='trainrgb', type=str,   help='path to the rgb txt file of trainset', required=True)
    parser.add_argument('--train-dep',    dest='traindep', type=str,   help='path to the depth txt file of trainset', required=True)
    parser.add_argument('--val-rgb',      dest='valrgb',   type=str,   help='path to the rgb txt file of valset', required=True)
    parser.add_argument('--val-dep',      dest='valdep',   type=str,   help='path to the depth txt file of valset', required=True)
    parser.add_argument('--test-rgb',     dest='testrgb',  type=str,   help='path to the rgb txt file of testset')
    parser.add_argument('--test-dep',     dest='testdep',  type=str,   help='path to the depth txt file of testset')
    parser.add_argument('--batch', '-b',                   type=int,   help='batch size of trainset (default: 8)', default=8)
    parser.add_argument('--batchval', '--bval',            type=int,   help='batch size of valset (default: 8)', default=8)
    parser.add_argument('--epochs',                        type=int,   help='number of epochs (default: 8)', default=50)
    # optimizer
    parser.add_argument('--optimizer', '-o',               type=str,   help='optimizer: '+'|'.join(opt_names)+' (default: sgd)', default='sgd', choices=opt_names)
    parser.add_argument('--lr',                            type=float, help='initial learning rate (default: 1e-2)', default=1e-4)
    parser.add_argument('--final-lr', '--flr', dest='flr', type=float, help='final learning rate of adabound (default: 1e-2)', default=1e-2)
    parser.add_argument('--momentum',                      type=float, help='momentum (default: 0.9)', default=0.9)
    parser.add_argument('--gamma',                         type=float, help='convergence speed term of AdaBound (default: 1e-3)' , default=1e-3)
    parser.add_argument('--beta1',                         type=float, help='Adam coefficients beta1 (default: 0.9)', default=0.9)
    parser.add_argument('--beta2',                         type=float, help='Adam coefficients beta2 (default: 0.999)', default=0.999)
    parser.add_argument('--weight-decay', dest='wd',       type=float, help='initial learning rate (default: 5e-4)', default=5e-4)
    # scheduler
    parser.add_argument('--scheduler', '-s',               type=str,   help='mode: '+'|'.join(sch_names)+' (default: step)', default='step',  choices=sch_names)
    parser.add_argument('--lr-decay', '--lrd', dest='lrd', type=float, help='lr decay rate of poly scheduler (default: 0.9)' , default=0.9)
    parser.add_argument('--alpha-seg', '-W',  dest='Wseg', type=float, help='coefficient of segmentation loss (default: 0.5)', default=0.5)
    parser.add_argument('--gpu',                           type=bool,  help='GPU or CPU (default: True)', default=True)
    parser.add_argument('--threads',                       type=int,   help='number of threads for data loading (default: 4)', default=4)
    parser.add_argument('--classes',                       type=int,   help='number of discrete classes of detph (default: 80)', default=80)
    parser.add_argument('--eval-freq', '-f', dest='f',     type=int,   help='number of evaluation interval during training (default: 1)', default=1)
    parser.add_argument('--workdir',                       type=str,   help='directory for storing training files', default='../workdir/')
    parser.add_argument('--logdir',                        type=str,   help='subdir of workdir, storing checkpoint and logfile (style: LOG_net_dataset_exp)')
    parser.add_argument('--resdir',                        type=str,   help='subdir of logdir, storing predicted results (default: res)', default='res')
    parser.add_argument('--resume',                                    help='reloaded checkpoint, absolute path (str), given epoch number (int) or nn.Module class')

    args = parser.parse_args()

    if args.dataset == 'nyu':
        args.min_depth, args.max_depth = 0.65, 10.0
    elif args.dataset == 'kitti':
        args.min_depth, args.max_depth = 1.80, 80.0
    return args

def compute_errors(gt, pred):
    """
    Args:
            gt, pred [batch, h, w]
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




def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:,:,:3] # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])
    
    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)