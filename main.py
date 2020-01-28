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
import os
import argparse
import time
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.backends import cudnn
import torchviz as tw
import matplotlib.pyplot as plt
from src import n2n
from src.custom_arch import *
from src.checkpoint_utils import makeSparse, genDenseModel
from src.group_lasso_regs import get_group_lasso_global, get_group_lasso_group
from src.utils import AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')

# Baseline
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=8, type=int, metavar='N',
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
parser.add_argument('--gpu_id', default='2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-s', '--numOfStages', default=3, type=int, help='defines the number of stages in the network')
parser.add_argument('-n', '--numOfBlocksinStage', type=int, default=5, help='defines the number of Blocks per Stage')
parser.add_argument('-l', '--layersInBlock', type=int, default=3, help='defines the number of')
# PruneTrain
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')
parser.add_argument('--sparse_interval', default=0, type=int,
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float,
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=False, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--global_group_lasso', default=False, action='store_true',
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

# N2N
parser.add_argument('--deeper', default=False, action='store_true',
                    help='Make network deeper')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.manual_seed(args.manualSeed)

best_acc = 0  # best test accuracy


def visualizePruneTrain(model, epoch):
    altList= []
    paramList = []
    printName = False
    for name, param in model.named_parameters():
        # print("\nName: {}", name)
        paramList.append(param)
        # print("\nName: ", name)
        i = int(name.split('.')[1])

        if i % 2 == 0:
            altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')
            if printName:
                print("\nI:", i, " ; ", altList[-1])
        elif (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
            if printName:
                print("\nI:", i, " ; ", altList[-1])
        elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")
            if printName:
                print("\nI:", i, " ; ", altList[-1])
        elif (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
            if printName:
                print("\nI:", i, " ; ", altList[-1])
        elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
            if printName:
                print("\nI:", i, " ; ", altList[-1])
        else:
            assert True, print("Hier fehlt noch was!!")
    # print("\naltList", altList)



    for i in range(0,len(altList)):
        if 'conv' in  altList[i]:
            paramList[i]=paramList[i].view(paramList[i].shape[0],paramList[i].shape[1],-1)
            weight = paramList[i].cpu()
            weight = weight.detach().numpy()
            plt.plot(  weight[0, ...]    )
            fileName = altList[i]+'_' + str(epoch)+ '.png'
            plt.savefig(fileName)


def main():
    # use anomaly detection of torch
    torch.autograd.set_detect_anomaly(True)

    global best_acc

    # Data
    # print('==> Preparing dataset %s' % args.dataset)
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

    # Model
    model = n2n.N2N(num_classes, args.numOfStages, args.numOfBlocksinStage, args.layersInBlock, True)
    model.cuda()
    # print(model)
    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Train and val
    # how many times N2N should make the network deeper

    start = time.time()
    for epochNet2Net in range(1, 2):

        best_acc = 0
        for epoch in range(1, args.epochs + 1):
            # adjust learning rate when epoch is the scheduled epoch
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))
            train_loss, train_acc, lasso_ratio, train_epoch_time = train(trainloader, model, criterion, optimizer,
                                                                         epoch, use_cuda)
            test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, epoch, use_cuda)
            tmp_parameters = list(model.parameters())
            # SparseTrain routine
            if args.en_group_lasso and (epoch % args.sparse_interval == 0):
                # Force weights under threshold to zero
                dense_chs, chs_map = makeSparse(optimizer, model, args.threshold,
                                                is_gating=args.is_gating)
                genDenseModel(model, dense_chs, optimizer, 'cifar')
                model = n2n.N2N(num_classes, args.numOfStages, args.numOfBlocksinStage, args.layersInBlock, False,
                                model)

                model.cuda()
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            visualizePruneTrain(model, epoch)
            best_acc = max(test_acc, best_acc)
            # print(model)
        print('Best acc:')
        print(best_acc)
        if (args.deeper):
            print("\n\nnow deeper")
            # deeper student training
            if best_acc < 50:
                model = n2n.deeper(model, optimizer, [2, 4])
            elif best_acc < 75:
                model = n2n.deeper(model, optimizer, [2])
            elif best_acc < 95:
                model = n2n.deeper(model, optimizer, [2])
            model.cuda()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)

    ende = time.time()
    print('{:5.3f}s'.format(ende - start), end='  ')
    print("\n")


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

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            inputs = Variable(inputs)
        targets = torch.autograd.Variable(targets)
        outputs = model.forward(inputs)
        # print("\n\nOutput Shape: ", outputs.shape)
        if batch_idx == 0:
            dot = tw.make_dot(outputs, params=dict(model.named_parameters()))
            filename = 'PruneTrain' + str(epoch) + '_' + str(batch_idx) + '.dot'
            dot.render(filename=filename)
        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1

        if args.en_group_lasso:
            if args.global_group_lasso:
                lasso_penalty = get_group_lasso_global(model)
            else:
                lasso_penalty = get_group_lasso_group(model)

            # Auto-tune the group-lasso coefficient @first training iteration
            coeff_dir = os.path.join(args.coeff_container, 'cifar')
            if init_batch:
                args.grp_lasso_coeff = args.var_group_lasso_coeff * loss.item() / (lasso_penalty *
                                                                                   (1 - args.var_group_lasso_coeff))
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
        # print("Loss: ", loss)
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
    return losses.avg, top1.avg, lasso_ratio.avg, epoch_time


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
            # targets = Variable(targets)
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


def adjust_learning_rate(optimizer, epoch):
    global state
    if args.schedule_exp == 0:
        # Step-wise LR decay
        set_lr = args.lr
        for lr_decay in args.schedule:
            if epoch >= lr_decay:
                set_lr *= args.gamma
        state['lr'] = set_lr
        args.lr = set_lr
    else:
        # Exponential LR decay
        set_lr = args.lr
        exp = int((epoch - 1) / args.schedule_exp)
        state['lr'] = set_lr * (args.gamma ** exp)
        args.lr = set_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
