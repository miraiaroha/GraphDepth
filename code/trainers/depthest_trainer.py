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
from torchstat import stat
import matplotlib.pyplot as plt


class DepthEstimationTrainer(Trainer):
    def __init__(self, params, net, datasets, criterion, optimizer, scheduler, 
                 sets=['train', 'val', 'test'], verbose=50, eval_func=None):
        self.params = params
        self.verbose = verbose
        self.eval_func = eval_func
        # Init dir
        if params.workdir is not None:
            workdir = os.path.expanduser(params.workdir)
        if params.logdir is None:
            logdir = os.path.join(workdir, 'log_{}_{}'.format(type(net).__name__, type(datasets[sets[0]]).__name__))
        else:
            logdir = os.path.join(workdir, params.logdir)
        resdir = None
        if self.datasets['test'] is not None:
            if params.resdir is None:
                resdir = os.path.join(logdir, '/res')
            else:
                resdir = os.path.join(logdir, params.resdir)
        # Call the constructor of the parent class (Trainer)
        super().__init__(net, datasets, optimizer, scheduler, criterion,
                         batch_size=params.b, batch_size_val=params.bval,
                         max_epochs=params.epochs, eval_freq=params.f,
                         gpus=params.gpus, resume=params.resume, mode=params.mode,
                         sets=sets, workdir=workdir, logdir=logdir, resdir=resdir)
        # uncomment to display the model complexity
        #stat(self.net, (3, self.params['height'], self.params['width']))

    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.logdir:
            self.writer = SummaryWriter(self.logdir)
        else:
            raise Exception("Log dir doesn't exist!")
        # Calculate total step
        self.n_train = len(self.trainset)
        self.steps_per_epoch = np.ceil(
            self.n_train / self.batch_size).astype(np.int32)
        self.verbose = min(self.verbose, self.steps_per_epoch)
        self.n_steps = self.max_epochs * self.steps_per_epoch
        # calculate model parameters memory
        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        memory = para * 4 / (1024**2)
        self.print('Model {} : params: {:4f}MB'.format(
            self.net._get_name(), memory))
        self.print('###### Experiment Parameters ######')
        for k, v in self.params.items():
            self.print('{0:<22s} : {1:}'.format(k, v))
        self.print("{0:<22s} : {1:} ".format('trainset sample', self.n_train))
        # GO!!!!!!!!!
        start_time = time.time()
        self.train_total_time = 0
        self.time_sofar = 0
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            # Decay Learning Rate
            self.scheduler.step()
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
            # Train one epoch
            total_loss = self.train_epoch(epoch)
            self.writer.add_scalar('loss', total_loss, self.global_step)
            torch.cuda.empty_cache()
            # Evaluate the model
            if self.eval_freq and epoch % self.eval_freq == 0:
                acc = self.eval(epoch)
                self.writer.add_scalar('acc', acc, epoch)
        self.print("Finished training! Best epoch {} best acc {:.4f}".format(
            self.best_epoch, self.best_acc))
        self.print("Spend time: {:.2f}h".format(
            (time.time() - start_time) / 3600))
        return

    def train_epoch(self, epoch):
        self.net.train()
        device = torch.device('cuda:0' if self.gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        # Iterate over data.
        prefetcher = data_prefetcher(self.trainloader)
        images, labels = prefetcher.next()
        step = 0
        while images is not None:
            images, labels = images.to(device), labels.to(device)
            before_op_time = time.time()
            self.optimizer.zero_grad()
            output = self.net(images)
            total_loss = self.criterion(output, labels)
            total_loss.backward()
            self.optimizer.step()
            fps = image.shape[0] / (time.time() - before_op_time)
            time_sofar = self.train_total_time / 3600
            time_left = (self.n_steps / self.global_step - 1.0) * time_sofar
            if self.verbose > 0 and (step + 1) % (self.steps_per_epoch // self.verbose) == 0:
                print_str = 'Epoch [{:>3}/{:>3}] | Step [{:>3}/{:>3}] | fps {:4.2f} | Loss: {:7.3f} | Time elapsed {:.2f}h | Time left {:.2f}h'. \
                    format(epoch, self.max_epochs, step + 1, self.steps_per_epoch, fps, total_loss, time_sofar, time_left)
                self.print(print_str)
            self.global_step += 1
            self.train_total_time += time.time() - before_op_time
            image, label = prefetcher.next()
            step += 1
        return total_loss

    def eval(self, epoch):
        torch.backends.cudnn.benchmark = True
        self.n_val = len(self.valset)
        self.print("{0:<22s} : {1:} ".format('valset sample', self.n_val))
        self.print("<-------------Evaluate the model-------------->")
        measure_list = ['a1', 'a2', 'a3', 'rmse', 'rmse_log', 'log10', 'abs_rel', 'sq_rel']
        # Evaluate one epoch
        measures_str, fps = self.eval_epoch(measure_list)
        acc = measures_str['a1']
        self.print('The {}th epoch, fps {:4.2f} | {}'.format(epoch, fps, measures_str))
        # Save the checkpoint
        if self.logdir:
            self.save_checkpoint(epoch, acc)
            self.print('=> Checkpoint was saved successfully!')
        else:
            if acc >= self.best_acc:
                self.best_epoch, self.acc = epoch, acc
        return acc

    def eval_epoch(self, measure_list):
        device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.net.to(device)
        self.criterion.to(device)
        self.net.eval()
        val_total_time = 0
        measures = {key: 0 for key in measure_list}
        with torch.no_grad():
            prefetcher = data_prefetcher(self.valloader)
            images, labels = prefetcher.next()
            step = 0
            while images is not None:
                images, labels = images.to(device), labels.to(device)
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
                # if step == 10:
                #     display_figure(params, myModel, criterion, writer,
                #                     images, depths, predict, epoch)
                print('Test step [{}/{}].'.format(step + 1,
                                                  len(self.valloader)))
                images, labels = prefetcher.next()
                step += 1
        measures_str = {key: '{:.5f}'.format(value / self.n_val) for key, value in measures.items()}
        fps = self.n_val / val_total_time
        return measures_str, fps

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
            images, labels = prefetcher.next()
            step = 0
            while images is not None:
                images, labels = images.to(device), labels.to(device)
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
                images, labels = prefetcher.next()
                step += 1
        measures_str = {key: '{:.5f}'.format(value / n_test) for key, value in measures.items()}
        fps = n_test / test_total_time
        self.print('Testing done, fps {:4.2f} | {}'.format(fps, measures_str))
        return


if __name__ == '__main__':
    pass
