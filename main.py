"""
 Copyright 2019 Sangkug Lym
 Copyright 2019 The University of Texas at Austin

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import copy
import os
import shutil
import time
import random

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import n2n
import src.src.models.cifar as models

from src.src.utils import AverageMeter, accuracy, mkdir_p, savefig
from src.src.custom import _makeSparse, _genDenseModel, _DataParallel
from src.src.custom import get_group_lasso_global, get_group_lasso_group
from src.src.custom_arch import *
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# Baseline
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--train_batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu-id', default='2', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# PruneTrain
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')
parser.add_argument('--sparse_interval', default=0, type=int,
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float,
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=True, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--global_group_lasso', default=True, action='store_true',
                    help='True: use a global group lasso coefficient, '
                         'False: use sqrt(num_params) as a coefficient for each group')
parser.add_argument('--var_group_lasso_coeff', default=0.1, type=float,
                    help='Ratio = group-lasso / (group-lasso + loss)')
parser.add_argument('--grp_lasso_coeff', default=0.0005, type=float,
                    help='claim as a global param')
parser.add_argument('--is_gating', default=False, action='store_true',
                    help='Use gating for residual network')
parser.add_argument('--threshold_type', default='max', choices=['max', 'mean'], type=str,
                    help='Thresholding type')
parser.add_argument('--coeff_container', default='./coeff', type=str,
                    help='Directory to store lasso coefficient')
parser.add_argument('--global_coeff', default=True, action='store_true',
                    help='Use a global group lasso regularizaiton coefficient')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
#N2N
parser.add_argument('--deeper', default=False, action='store_true',
                    help='Male network deeper')



args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    torch.autograd.set_detect_anomaly(True)
    global best_acc
    # Data
    #print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataloader = datasets.CIFAR10
    num_classes = 10

    trainset = dataloader(root='./dataset/data/torch', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset,
                                  batch_size=args.train_batch,
                                  shuffle=True,
                                  num_workers=args.workers)

    testset = dataloader(root='./dataset/data/torch', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    model = n2n.N2N(num_classes)
    model.cuda()

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train and val
    for epochNet2Net in range(1, 3):
        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch)

            #print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

            train_loss, train_acc, lasso_ratio, train_epoch_time = train(trainloader, model, criterion, optimizer,
                                                                         epoch, use_cuda)
            test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, epoch, use_cuda)

            # SparseTrain routine
            if args.en_group_lasso and (epoch % args.sparse_interval == 0):
                # Force weights under threshold to zero
                dense_chs, chs_map = _makeSparse(model, args.threshold,
                                                 is_gating=args.is_gating)
                # Reconstruct architecture
                _genDenseModel(model, dense_chs, optimizer, 'cifar')

            best_acc = max(test_acc, best_acc)
        print('Best acc:')
        print(best_acc)
        if(args.deeper):
            print("\n\nnow deeper")
            # deeper student training
            if best_acc< 50:
                model = model.deeper(model, [2,8])
            elif best_acc < 75:
                model = model.deeper(model, [2,6])
            elif best_acc < 95:
                model = model.deeper(model, [6])
        model.cuda()


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lasso_ratio = AverageMeter()

    end = time.time()
    input_size = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end
        input_size = inputs.size
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()


        #inputs = Variable(inputs)
        #target = torch.autograd.Variable(targets)

        with torch.no_grad():
            inputs = Variable(inputs)
        targets = torch.autograd.Variable(targets)
        outputs = model.forward(inputs)

        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1

        if args.en_group_lasso:
            if args.global_group_lasso:
                lasso_penalty = get_group_lasso_global(model)
            else:
                lasso_penalty = get_group_lasso_group(model)

            # Auto-tune the group-lasso coefficient @first training iteration
            coeff_dir = os.path.join(args.coeff_container)
            if init_batch:
                args.grp_lasso_coeff = args.var_group_lasso_coeff * loss.item() / (
                        lasso_penalty * (1 - args.var_group_lasso_coeff))
                grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)

                if not os.path.exists(coeff_dir):
                    os.makedirs(coeff_dir)
                with open(os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'w') as f_coeff:
                    f_coeff.write(str(grp_lasso_coeff.item()))

            else:
                with open(os.path.join(coeff_dir, str(args.var_group_lasso_coeff)), 'r') as f_coeff:
                    for line in f_coeff:
                        grp_lasso_coeff = float(line)

            lasso_penalty = lasso_penalty * grp_lasso_coeff
        else:
            lasso_penalty = 0.

        # Group lasso calcution is not performance-optimized => Ignore from execution time
        loss += lasso_penalty

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()

        if batch_idx % args.print_freq == 0:
             print('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                 epoch, batch_idx, len(trainloader), batch_time=batch_time,
                 data_time=data_time, loss=losses, top1=top1, top5=top5))

    epoch_time = batch_time.avg * len(trainloader)  # Time for total training dataset
    return (losses.avg, top1.avg, lasso_ratio.avg, epoch_time)


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            #targets = Variable(targets)
        targets = torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()

    epoch_time = batch_time.avg * len(testloader)  # Time for total test dataset
    return (losses.avg, top1.avg, epoch_time)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if args.schedule_exp == 0:
        # Step-wise LR decay
        set_lr = args.lr
        for lr_decay in args.schedule:
            if epoch >= lr_decay:
                set_lr *= args.gamma
        state['lr'] = set_lr
    else:
        # Exponential LR decay
        set_lr = args.lr
        exp = int((epoch - 1) / args.schedule_exp)
        state['lr'] = set_lr * (args.gamma ** exp)

    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
