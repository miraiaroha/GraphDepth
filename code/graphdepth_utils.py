import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse

cmap = plt.cm.viridis

def load_params_from_parser():
    modes = ['train', 'eval', 'test', 'finetune']
    encoder_names = ['resnet50', 'resnet101']
    decoder_names = ['graph']
    loss_names = ['l1', 'l2']
    opt_names = ['sgd', 'adam']
    data_names = ['nyu', 'kitti']
    # from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
    # sparsifier_names = [x.name for x in [UniformSampling, SimulatedStereo]]
    # from models import Decoder
    # decoder_names = Decoder.names
    # from dataloaders.dataloader import MyDataloader
    # modality_names = MyDataloader.modality_names
    parser = argparse.ArgumentParser(description='GraphDepth Pytorch Implementation.')
    parser.add_argument('--mode',                     type=str,   help='mode: '+'|'.join(modes)+' (default: fintune)', default='finetune',  choices=modes)
    parser.add_argument('--encoder',                  type=str,   help='encoder: '+'|'.join(encoder_names)+' (default: resnet50)', default='resnet50', choices=encoder_names)
    parser.add_argument('--decoder',                  type=str,   help='decoder: '+'|'.join(decoder_names)+' (default: graph', default='graph', choices=decoder_names)
    parser.add_argument('--dataset',                  type=str,   help='dataset: '+'|'.join(data_names)+' (default: nyu)', default='nyu', choices=data_names)
    parser.add_argument('--train-path', '--trainp',   type=str,   help='path to the trainset', required=True)
    parser.add_argument('--val-path', '--valp'        type=str,   help='path to the valset', required=True)
    parser.add_argument('--test-path', '--testp'      type=str,   help='path to the testset')
    parser.add_argument('--train-txt', '--traint'     type=str,   help='path to the trainset txt file', required=True)
    parser.add_argument('--val-txt', '--valt'         type=str,   help='path to the valset txt file', required=True)
    parser.add_argument('--test-txt', '--testt'       type=str,   help='path to the testset txt file')
    parser.add_argument('--batch-size', '-b'          type=int,   help='batch size of trainset (default: 8)', default=8)
    parser.add_argument('--batch-size-val', '--bval', type=int,   help='batch size of valset (default: 8)', default=8)
    parser.add_argument('--epochs',                   type=int,   help='number of epochs (default: 8)', default=50)
    parser.add_argument('--lr',                       type=float, help='initial learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--optimizer', '-o'                       help='optimizer: '+'|'.join(opt_names)+' (default: sgd)', default='sgd', choices=opt_names)
    parser.add_argument('--momentum',                 type=float, help='momentum (default: 0.9)', default=0.9)
    parser.add_argument('--weight-decay', '--wd',     type=float, help='initial learning rate (default: 1e-4)', default=1e-4)
    parser.add_argument('--alpha-seg', '--Wseg',      type=float, help='coefficient of segmentation loss (default: 0.5)', default=0.5)
    parser.add_argument('--gpus',                     type=int,   help='number of GPUs (default: 1)', default=1)
    parser.add_argument('--threads',                  type=int,   help='number of threads for data loading (default: 4)', default=4)
    parser.add_argument('--classes',                  type=int,   help='number of discrete classes of detph (default: 80)', default=80)
    parser.add_argument('--eval-freq', '-f'           type=int,   help='number of evaluation interval during training (default: 1)', default=1)
    parser.add_argument('--workdir'                   type=str,   help='directory for storing training files', default='../workspace/')
    parser.add_argument('--logdir',                   type=str,   help='subdir of workdir, storing checkpoint and logfile (style: /enc_dec_dataset_exp)')
    parser.add_argument('--resume',                               help='reloaded checkpoint, absolute path (str), given epoch number (int)')

    args = parser.parse_args()

    if args.dataset == 'nyu':
        args.min_depth, args.max_depth = 0.65, 10.0
    elif args.dataset == 'kitti':
        args.min_depth, args.max_depth = 1.80, 80.0
    return args

def save_checkpoint(state, is_best, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch) + '.pth.tar')
    torch.save(state, checkpoint_filename)
    if is_best:
        best_filename = os.path.join(output_directory, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)
    if epoch > 0:
        prev_checkpoint_filename = os.path.join(output_directory, 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)

def adjust_learning_rate(optimizer, epoch, lr_init, lr_decay_step):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    lr = lr_init * (0.1 ** (epoch // lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_output_directory(args):
    output_directory = os.path.join('results',
        '{}.samples={}.modality={}.arch={}.criterion={}.lr={}.bs={}'.
        format(args.data, args.num_samples, args.modality, \
            args.arch, args.criterion, args.lr, args.batch_size))
    return output_directory


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