import os
from torch import nn
from models.mobilefacenet import MobileFacenet
from dataloader.CASIA_Face_loader import CASIA_Face
from dataloader.LFW_loader import LFW
from dataloader.HyperECUST_loader import HyperECUST_FV, HyperECUST_FI
from trainers.faceverification_trainer import MobileFacenetTrainer
from utils.faceverification_utils import Evaluation_10_fold
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
# gpu init
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['GPU_DEBUG'] = '0'
# -----------------------------------------------------
CASIA_path = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96'
CASIA_txt = '~/yrc/myFile/sphereface/train/data/CASIA-WebFace-112X96.txt'
LFW_path = '~/yrc/myFile/sphereface/test/data/lfw-112X96'
LFW_txt = '~/yrc/myFile/sphereface/test/data/pairs.txt'
ECUST_path = '~/myDataset/ECUST_112x96'


def main(split=13, fold=0, s=0):
    params = {
        'trainset_path': ECUST_path,
        'trainset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/train_fold_{}.txt'.format(split, fold),
        'validset_path': ECUST_path,
        'validset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/pairs_fold_{}.txt'.format(split, fold),
        # '~/yrc/myFile/sphereface/test/data/pairs.txt',
        # '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verfication_split/split_1/pairs.txt',
        'testset_path': ECUST_path,
        'testset_txt': '~/yrc/myFile/huaweiproj/code/cyr/datasets/face_verification_split/split_{}/pairs_fold_{}.txt'.format(split, fold),
        #'~/yrc/myFile/huaweiproj/code/datasets/cyr/face_verfication_split/split_{}/pairs_exp_{}.txt'.format(split, 0),
        'workspace_dir': '~/yrc/myFile/huaweiproj/code/cyr/workspace/112x96_noise_{}'.format(s),
        'log_dir': 'MobileFacenet_HyperECUST_fold_{}'.format(fold),
        #'MobileFacenet_split_{}_exp_{}'.format(split, 0),
        'batch_size': 256,
        'batch_size_valid': 256,
        'max_epochs': 20,
        'num_classes': 33,
        'use_gpu': True,
        'height': 112,
        'width': 96,
        'test_freq': 1,
        'resume': './workspace/112x96/MobileFacenet_HyperECUST_fold_{}/MobileFacenet_best.pkl'.format(fold),
        #'./workspace/MobileFacenet_CASIA_Face/MobileFacenet_best.pkl'
        #'./workspace/' + resolution + '/MobileFacenet/MobileFacenet_best.pkl'
        #'./workspace/MobileFacenet_split_{}_exp_{}/MobileFacenet_best.pkl'.format(split, 0)
    }
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
    # Trainer.train()
    # return

    # Evaluate one epoch
    Trainer.reload()
    acc, threshold, _ = Trainer.eval_epoch(filename='valid_result.mat')
    print("acc {:.4f}, threshold {:.4f}".format(acc, threshold))
    return

    # Test model
    # Trainer.load_checkpoint(params['resume'])
    # Trainer.test(0.4)
    # return

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


# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    # main()
    for s in [0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.04, 0.06]:
        for i in range(5):
            print('---------fold_{}_snr_{}---------'.format(i, s))
            main(fold=i, s=s)
