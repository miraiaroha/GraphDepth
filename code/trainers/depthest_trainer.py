import os
import sys
import time
import datetime
import numpy as np
import scipy.io
sys.path.append(os.path.dirname(__file__))
from tensorboardX import SummaryWriter
from trainer import Trainer, data_prefetcher
import torch
from torch.nn import DataParallel
import matplotlib.pyplot as plt
from copy import deepcopy

class DepthEstimationTrainer(Trainer):
    def __init__(self, params, net, datasets, criterion, optimizer, scheduler, 
                 sets=['train', 'val', 'test'], verbose=50, stat=False, 
                 eval_func=None, disp_func=None):
        self.params = params
        self.verbose = verbose
        self.eval_func = eval_func
        self.disp_func = disp_func
        # Init dir
        if params.workdir is not None:
            workdir = os.path.expanduser(params.workdir)
        logdir = None
        if params.logdir is None:
            logdir = os.path.join(workdir, 'LOG_{}_{}'.format(params.encoder+params.decoder, params.dataset))
        else:
            logdir = os.path.join(workdir, params.logdir)
        self.params.logdir = logdir
        resdir = None
        if datasets['test'] is not None:
            if params.resdir is None:
                resdir = os.path.join(logdir, 'res')
            else:
                resdir = os.path.join(logdir, params.resdir)
        self.params.resdir = resdir
        # Call the constructor of the parent class (Trainer)
        super().__init__(net, datasets, optimizer, scheduler, criterion,
                         batch_size=params.batch, batch_size_val=params.batchval,
                         max_epochs=params.epochs, threads=params.threads, eval_freq=params.f,
                         use_gpu=params.gpu, resume=params.resume, mode=params.mode,
                         sets=sets, workdir=workdir, logdir=logdir, resdir=resdir)
        # uncomment to display the model complexity
        #print(self.net.conv1.__class__.__call__)
        if stat:
            from torchstat import stat
            net_copy = deepcopy(self.net)
            print(id(self.net), id(net_copy))
            stat(net_copy, (3, *self.datasets[sets[0]].input_size))
            #del net_copy
            #exit()
        #print(self.net.conv1.__class__.__call__)
        #print(net_copy.conv1.__class__.__call__)
        #exit()

    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.logdir:
            self.writer = SummaryWriter(self.logdir)
        else:
            raise Exception("Log dir doesn't exist!")
        # Calculate total step
        self.n_train = len(self.trainset)
        self.steps_per_epoch = np.ceil(self.n_train / self.batch_size).astype(np.int32)
        self.verbose = min(self.verbose, self.steps_per_epoch)
        self.n_steps = self.max_epochs * self.steps_per_epoch
        # calculate model parameters memory
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        memory = para * 4 / (1024**2)
        self.print('Model {} : params: {:,}, Memory {:.3f}MB'.format(self.net._get_name(), para, memory))
        self.print('###### Experiment Parameters ######')
        for k, v in vars(self.params).items():
            self.print('{0:<22s} : {1:}'.format(k, v))
        self.print("{0:<22s} : {1:} ".format('trainset sample', self.n_train))
        # GO!!!!!!!!!
        start_time = time.time()
        self.train_total_time = 0
        self.time_sofar = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            torch.cuda.empty_cache()
            # Decay Learning Rate
            if self.params.scheduler in ['step', 'plateau']:
                self.scheduler.step()
            # Evaluate the model
            if self.eval_freq and epoch % self.eval_freq == 0:
                measures = self.eval(epoch)
                for k in sorted(list(measures.keys())):
                    self.writer.add_scalar(k, measures[k], epoch)
        self.print("Finished training! Best epoch {} best acc {:.4f}".format(self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format((time.time() - start_time) / 3600))
        return

    def train_epoch(self, epoch):
        self.net.train()
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        # Iterate over data.
        prefetcher = data_prefetcher(self.trainloader)
        data = prefetcher.next()
        step = 0
        while data is not None:
            images, labels = data[0].to(device), data[1].to(device)
            before_op_time = time.time()
            self.optimizer.zero_grad()
            output = self.net(images)
            total_loss = self.criterion(output, labels.squeeze())
            total_loss.backward()
            self.optimizer.step()
            fps = images.shape[0] / (time.time() - before_op_time)
            time_sofar = self.train_total_time / 3600
            time_left = (self.n_steps / self.global_step - 1.0) * time_sofar
            lr = self.optimizer.param_groups[0]['lr']
            if self.verbose > 0 and (step + 1) % (self.steps_per_epoch // self.verbose) == 0:
                print_str = 'Epoch[{:>2}/{:>2}] | Step[{:>4}/{:>4}] | fps {:4.2f} | Loss: {:7.3f} | elapsed {:.2f}h | left {:.2f}h | lr {:.3e}'. \
                    format(epoch, self.max_epochs, step + 1, self.steps_per_epoch, fps, total_loss, time_sofar, time_left, lr)
                self.print(print_str)
            self.writer.add_scalar('loss', total_loss, self.global_step)
            self.writer.add_scalar('lr', lr, epoch)
            # Decay Learning Rate
            if self.params.scheduler == 'poly':
                self.scheduler.step()
            self.global_step += 1
            self.train_total_time += time.time() - before_op_time
            data = prefetcher.next()
            step += 1
        return total_loss

    def eval(self, epoch):
        torch.backends.cudnn.benchmark = True
        self.n_val = len(self.valset)
        self.print("{0:<22s} : {1:} ".format('valset sample', self.n_val))
        self.print("<-------------Evaluate the model-------------->")
        # Evaluate one epoch
        measures, fps = self.eval_epoch(epoch)
        acc = measures['a1']
        measures = {key: round(value/self.n_val, 5) for key, value in measures.items()}
        self.print('The {}th epoch, fps {:4.2f} | {}'.format(epoch, fps, measures))
        # Save the checkpoint
        if self.logdir:
            self.save(epoch, acc)
        else:
            if acc >= self.best_acc:
                self.best_epoch, self.acc = epoch, acc
        return measures

    def eval_epoch(self, epoch):
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.eval()
        val_total_time = 0
        measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        measures = dict(zip(measure_list, np.zeros(len(measure_list))))
        with torch.no_grad():
            prefetcher = data_prefetcher(self.valloader)
            data = prefetcher.next()
            step = 0
            while data is not None:
                images, labels = data[0].to(device), data[1].to(device)
                # forward
                before_op_time = time.time()
                output = self.net(images)
                depths = self.net.inference(output)
                duration = time.time() - before_op_time
                val_total_time += duration
                # accuracy
                new = self.eval_func(labels, depths)
                for i in range(len(measure_list)):
                    measures[measure_list[i]] += new[measure_list[i]].item()
                # display images
                if step == 10 and self.disp_func is not None:
                    self.disp_func(self.params, self.writer, self.net, images, labels, depths, epoch)
                print('Test step [{}/{}].'.format(step + 1, len(self.valloader)))
                data = prefetcher.next()
                step += 1
        
        fps = self.n_val / val_total_time
        return measures, fps

    def test(self):
        torch.backends.cudnn.benchmark = True
        n_test = len(self.testset)
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.net.eval()
        print("<-------------Test the model-------------->")
        colormap = {'nyu': 'jet', 'kitti': 'plasma'}
        measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        measures = {key: 0 for key in measure_list}
        test_total_time = 0
        with torch.no_grad():
            prefetcher = data_prefetcher(self.testloader)
            data = prefetcher.next()
            step = 0
            while data is not None:
                images, labels = data[0].to(device), data[1].to(device)
                # forward
                before_op_time = time.time()
                output = self.net(images)
                depths = self.net.inference(output)
                duration = time.time() - before_op_time
                test_total_time += duration
                # accuracy
                new = self.eval_func(labels, depths)
                print("Test step [{}/{}].".format(step + 1, n_test))
                images = np.transpose(images.cpu().numpy().squeeze(), (1, 2, 0))
                labels = (labels.cpu() / self.params.max_depth).numpy().squeeze()
                depths = (depths.cpu() / self.params.max_depth).numpy().squeeze()
                plt.imsave(os.path.join(self.resdir, '{:04}_rgb.png'.format(step)), images)
                plt.imsave(os.path.join(self.resdir, '{:04}_gt.png'.format(step)), labels, cmap=colormap[self.params.dataset])
                plt.imsave(os.path.join(self.resdir, '{:04}_depth.png'.format(step)), depths, cmap=colormap[self.params.dataset])

                for i in range(len(measure_list)):
                    measures[measure_list[i]] += new[measure_list[i]].item()
                data = prefetcher.next()
                step += 1
        measures_str = {key: '{:.5f}'.format(value / n_test) for key, value in measures.items()}
        fps = n_test / test_total_time
        self.print('Testing done, fps {:4.2f} | {}'.format(fps, measures_str))
        return


if __name__ == '__main__':
    pass
