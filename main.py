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
import copy
import os
import argparse
import shutil
import time
import random

import matplotlib
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.lars import LARS

from src.mem_reporter import *
from copy import deepcopy

from matplotlib import pyplot
from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d import Axes3D
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
from src.utils import AverageMeter, accuracy, mkdir_p, Logger
# from apex.parallel import DistributedDataParallel as DDP
# from apex.apex.fp16_utils import *
# from apex import amp, optimizers
# from apex.apex.multi_tensor_apply import multi_tensor_applier
import platform, psutil

# Parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# parameters for basic cuda
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--O1', default=False, action='store_true',
                    help='Use half precision apex methods O1')
parser.add_argument('--O2', default=False, action='store_true',
                    help='Use half precision apex methods O2')
parser.add_argument('--O3', default=False, action='store_true',
                    help='Use half precision apex methods O3')


# model size
parser.add_argument('-s', '--numOfStages', default=3, type=int, help='defines the number of stages in the network')
# parser.add_argument('-n', '--numOfBlocksinStage', type=int, default=5, help='defines the number of Blocks per Stage')
parser.add_argument('-l', '--layersInBlock', type=int, default=3, help='defines the number of')
parser.add_argument('-n', type=str, help="#stage numbers separated by commas")
parser.add_argument('-b', '--bottleneck', default=False, action='store_true',
                    help='Set the bootleneck parameter')
parser.add_argument('-w', '--widthofFirstLayer', default=3, type=int, help='defines the number of stages in the network')


# epochs and stuff
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochsFromBegin', default=0, type=int, metavar='N',
                    help='number of Epochs from begin (default: 0)')
parser.add_argument('--lastEpoch', default=False, action='store_true',
                    help='Last Epoch')
parser.add_argument('--test', default=False, action='store_true',
                    help='Should the Test run?')

# folder stuff
parser.add_argument('--pathToModell', type=str, help='Path to the location where the model will be stored')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--save_checkpoint', default=10, type=int,
                    help='Interval to save checkpoint')
parser.add_argument('--coeff_container', default='./coeff', type=str,
                    help='Directory to store lasso coefficient')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--saveModell', default=False, action='store_true',
                    help='Save Modell')

# batch size
parser.add_argument('--test_batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--batchTrue', default=False, action='store_true',
                    help='Set the batchsize')
parser.add_argument('--batch_size', default=1000, type=int,
                    metavar='N', help='batch size')

# other model stuff
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-dlr', '--delta_learning_rate', default=False, action='store_true',
                    help='No change in learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[93, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dynlr', default=False, action='store_true',
                    help='Dynamische LR')
parser.add_argument('--deltalr', default=False, action='store_true',
                    help='Verändere die LR')
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')

# PruneTrain
parser.add_argument('--sparse_interval', default=0, type=int,
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float,
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=False, action='store_true',
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
parser.add_argument('--global_coeff', default=True, action='store_true',
                    help='Use a global group lasso regularizaiton coefficient')
# divers
parser.add_argument('--print-freq', default=64, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cifar10', default=False, action='store_true',
                    help='Set the batchsize')
parser.add_argument('--cifar100', default=False, action='store_true',
                    help='Set the batchsize')
parser.add_argument('--visual', default=False, action='store_true',
                    help='Set the visual')

# N2N
parser.add_argument('--deeper', default=False, action='store_true',
                    help='Make network deeper')
# lars
parser.add_argument('--lars', default=False, action='store_true',
                    help='use lars')
parser.add_argument('--larsLR', default=0.001, type=float,
                    help='')


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

listofBlocks = [int(i) for i in args.n.split(',')]
print(listofBlocks)
# dev = "cuda:0"
# device = torch.device(dev)

best_acc = 0  # best test accuracy


def main():
    global best_acc

    # checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    # torch
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # choose which gpu to use
    # not_enough_memory = True
    # use_gpu = 'cuda:1'
    # use_gpu_num = 1
    # cudaArray = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    use_cuda = torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    # choose gpu
    # if int(args.gpu_id) < 5:
    #     gpu_id = args.gpu_id
    #     not_enough_memory = False
    #     use_gpu_num = int(gpu_id)
    # while not_enough_memory:
    #     if args.gpu1080:
    #         print(f'Nutze Geforce 1080')
    #         gpu_id = 1
    #     else:
    #         print(f'gpu id:2')
    #         gpu_id = 2
    #     print(f'Device Name: {torch.cuda.get_device_name(gpu_id)}')
    #     total, used, free = checkmem(gpu_id)
    #     if used<20:
    #         use_gpu = cudaArray[gpu_id]
    #         torch.cuda.set_device(gpu_id)
    #         use_gpu_num = gpu_id
    #         print(f'This Gpu is free')
    #         print(f'GPU Id: {gpu_id}')
    #         print(f'total    : {total}')
    #         print(f'free     : {free}')
    #         print(f'used     : {used}')
    #         print('\n')
    #         not_enough_memory = False
    #         break
    #
    #     if args.gpu1080 and not_enough_memory:
    #         print(f'GPU1080 id 3')
    #         gpu_id = 3
    #     elif not_enough_memory:
    #         print(f'GPU id:2')
    #         gpu_id = 2
    #     print(f'Device Name: {torch.cuda.get_device_name(gpu_id)}')
    #     total, used, free = checkmem(gpu_id)
    #     if used < 20:
    #         use_gpu = cudaArray[gpu_id]
    #         use_gpu_num = gpu_id
    #         torch.cuda.set_device(gpu_id)
    #         print(f'This Gpu is free')
    #         print(f'GPU Id: {gpu_id}')
    #         print(f'total    : {total}')
    #         print(f'free     : {free}')
    #         print(f'used     : {used}')
    #         print('\n')
    #         not_enough_memory = False
    #         break
    #     if not_enough_memory:
    #         time.sleep(600)

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.manual_seed(args.manualSeed)

    # use anomaly detection of torch
    torch.autograd.set_detect_anomaly(True)

    # Transform Train and Test data

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

    # Load data
    # print(f'Cifar10: {args.cifar10}; cifar100: {args.cifar100}')
    if args.cifar10 and not args.cifar100:
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.cifar100 and not args.cifar10:
        dataloader = datasets.CIFAR100
        num_classes = 100
    dataset = not ((not args.cifar10 and not args.cifar100) or (args.cifar10 and args.cifar100))
    assert dataset, "kein gültiger Datensatz angegeben"

    trainset = dataloader(root='./dataset/data/torch', train=True, download=True, transform=transform_train)

    testset = dataloader(root='./dataset/data/torch', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # dynamic resnet modell

    title = 'prune' + str(args.epochsFromBegin)
    if args.resume:
        model = torch.load(args.pathToModell)
        model.cuda()
        # print(f'Model: {model}')
        criterion = nn.CrossEntropyLoss()
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        # start_batchSize = checkpoint['start_batchSize']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(
            ['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc.', 'ValidAcc.', 'TrainEpochTime(s)',
             'TestEpochTime(s)'])
        assert args.numOfStages == len(
            listofBlocks), 'Liste der Blöcke pro Stage sollte genauso lang sein wie Stages vorkommen!!!'
        model = n2n.N2N(num_classes, args.numOfStages, listofBlocks, args.layersInBlock, True, args.bottleneck, args.widthofFirstLayer )
        print(f'device: {torch.cuda.current_device()}')
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        start_epoch = 1

    print(f'Startepoche: {start_epoch}')

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # if args.O1:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # if args.O2:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    # if args.O3:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O3")

    # Count the parameters of the model and calculate training bacth size
    count0 = 0
    for p in model.parameters():
        count0 += p.data.nelement()
    print(f'count0: {count0}')
    count1 = count0

    # Calculate Size of Trainings Batch size
    if not args.batchTrue:

        # calculate first how many blocks is equal to the count0
        sizeX = (count0 - 1306) / 97216
        print(f'sizeX: {sizeX}')
        # Gerade für niedrige Batch size
        y = 36.304 * sizeX + 107.768

        # y = 4.27*sizeX + 2.60
        # calculate now the batch size
        batch_size = int(0.98 * count0 / sizeX / y)
        delta_bs = (batch_size - 330)*0.3
        batch_size = int(batch_size - delta_bs)
        print(f'batch_size: {batch_size};{y} ; lr: {args.lr}')

        args.batch_size = batch_size
    else:
        batch_size = args.batch_size
    args.lr *= (batch_size / 256)

    if not args.resume:
        start_batchSize = batch_size

    trainloader = data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                  shuffle=True, num_workers=args.workers)
    if not args.lars:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = LARS(model.parameters(),eta=args.larsLR, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    i = 1
    # for epochNet2Net in range(1, 4):
    while i == 1:
        for epoch in range(start_epoch, args.epochs + start_epoch):
            # adjust learning rate when epoch is the scheduled epoch
            if not args.delta_learning_rate:
                adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs + start_epoch - 1, args.lr))
            start = time.time()
            train_loss, train_acc, train_epoch_time = train(trainloader, model, criterion,
                                                            optimizer, epoch, use_cuda)
            ende = time.time()

            if args.test:
                test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc, train_epoch_time,
                           test_epoch_time])

            # i = 2
            # SparseTrain routine
            if not args.en_group_lasso:
                pass
            elif args.en_group_lasso and (epoch % args.sparse_interval == 0) and not args.lastEpoch:
                # Force weights under threshold to zero
                dense_chs, chs_map = makeSparse(optimizer, model, args.threshold)
                if args.visual:
                    visualizePruneTrain(model, epoch, args.threshold)

                genDenseModel(model, dense_chs, optimizer, 'cifar')
                model = n2n.N2N(num_classes, args.numOfStages, listofBlocks, args.layersInBlock, False, False, model,
                                model.archNums)
                # use_after_model_creation = torch.cuda.memory_allocated(use_gpu)
                # print(f'use after new Model Creation')
                gc.collect()
                model.cuda()
                if not args.lars:
                    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
                else:
                    optimizer = LARS(model.parameters(),eta=args.larsLR, lr=args.lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
            #     if args.fp16:
            #         model, optimizer = amp.initialize(model, optimizer)
            #
            count = 0
            for p in model.parameters():
                count += p.data.nelement()
            if count < count1:
                print(f'Count: {count} ; {count0} ; {count / count0}')
                count1 = count
            #     if (count/count0) > 0.9:
            #         a = 0.9
            #     elif (count/count0) > 0.7:
            #         a=0.8
            #
            #     else:
            #         a= 0.6
            #     y = int(m * args.numOfBlocksinStage*(count - m)/(count0 - m)+y0)
            #     print(f'Y: {y}')
            #     batch_size = int(count/ args.numOfBlocksinStage * 1 / y)
            #
            #     trainloader = data.DataLoader(trainset, batch_size=batch_size,
            #                                  shuffle=True, num_workers=args.workers)
            #
            #     print(f'new batch_size: {batch_size}')
            # # print("\nEpoche: ", epoch, " ; NumbOfParameters: ", count)
            #

            if args.deeper:
                print("\n\nnow deeper")
                # deeper student training
                model = n2n.deeper(model, optimizer, [2, 4])
                model.cuda()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            # print("\n Verhältnis Modell Größe: ", count / count0)

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            print("[INFO] Storing checkpoint...")
            save_checkpoint({
                'epoch': args.epochs + start_epoch,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),},
                is_best,
                checkpoint=args.checkpoint)
            # Leave unique checkpoint of pruned models druing training
            if epoch % args.save_checkpoint == 0:
                save_checkpoint({
                    'epoch': args.epochs + start_epoch - 1,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(), },
                    is_best,
                    checkpoint=args.checkpoint,
                    filename='checkpoint' + str(epoch) + '.tar')

        i = 2
    if args.saveModell:
        torch.save(model, args.pathToModell)
    logger.close()
    print("\n ", args.batch_size)  # , " ; ", args.numOfStages, " ; ", args.numOfBlocksinStage, " ; ", args.layersInBlock," ; ", args.epochs)
    if args.test:
        print(" ", test_acc)
    print(f'Max memory: {torch.cuda.max_memory_allocated() / 10000000}')
    print(' {:5.3f}s'.format(ende - start), end='  ')


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode

    model.train()

    # global grp_lasso_coeff
    # Measure time
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    lasso_ratio = AverageMeter()

    printLasso = False
    end = time.time()

    # for param in model.parameters():
    #    param.grad = None
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
        loss = criterion(outputs, targets)
        if printLasso:
            print(f'Loss: {loss}')

            # if batch_idx == 0 and (epoch % args.sparse_interval == 0):
            #     dot = tw.make_dot(outputs, params=dict(model.named_parameters()))
            #     filename = 'model/PruneTrain' + str(epoch) + '_' + str(batch_idx) + '.dot'
            #     dot.render(filename=filename)

            # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1

        if args.en_group_lasso:
            if args.global_group_lasso:
                lasso_penalty = get_group_lasso_global(model)
            else:
                lasso_penalty = get_group_lasso_group(model)
            if printLasso:
                print(f'Lasso Penalty1: {lasso_penalty}')
                # Auto-tune the group-lasso coefficient @first training iteration
                # coeff_dir = os.path.join(args.coeff_container, 'cifar')
            if init_batch:
                args.grp_lasso_coeff = args.var_group_lasso_coeff * loss.item() / (
                            lasso_penalty * (1 - args.var_group_lasso_coeff))
                grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)

                with open(os.path.join(args.checkpoint, str(args.var_group_lasso_coeff)), 'w') as f_coeff:
                    f_coeff.write(str(grp_lasso_coeff.item()))
                if printLasso:
                    print(f'Grp lasso coeff: {grp_lasso_coeff}')

            else:
                with open(os.path.join(args.checkpoint, str(args.var_group_lasso_coeff)), 'r') as f_coeff:
                    for line in f_coeff:
                        grp_lasso_coeff = float(line)
            lasso_penalty = lasso_penalty * grp_lasso_coeff
            if printLasso:
                print(f'Lasso Penalty2: {lasso_penalty}')
        else:
            lasso_penalty = 0
            # print(f'nach group lasso')
            # Group lasso calcution is not performance-optimized => Ignore from execution time
        loss += lasso_penalty
        # measure accuracy and record loss

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        optimizer.zero_grad()
        #   compute gradient and do SGD step
        if args.O1 or args.O2 or args.O3:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            # print(f'After backward')
        optimizer.step()

        # print(f'After Step')
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
    return losses.avg, top1.avg, epoch_time


def test(testloader, model, criterion, epoch, use_cuda):
    # print(f'Test')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # print(f'Vor der For Schleife Test mit Länge: {len(testloader)}')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # print(f'For Schleife betretten')
        # measure data loading time
        data_time.update(time.time() - end)
        # print(f'Time 1')
        data_load_time = time.time() - end
        # print(f'Test Variablen; use cuda: {use_cuda}')
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # print(f'Nach if use cuda')
        with torch.no_grad():
            inputs = Variable(inputs)
            # targets = Variable(targets)
        # print(f'vor target Variablen')
        targets = torch.autograd.Variable(targets)
        # compute output
        # print(f'Test vor dem Forward')

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # print(f'Test nachdem loss')
        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end - data_load_time)
        end = time.time()
    # print(f'Test Ende')
    epoch_time = batch_time.avg * len(testloader)  # Time for total test dataset
    return (losses.avg, top1.avg, epoch_time)


def adjust_learning_rate(optimizer, epoch):
    global state
    if args.dynlr and args.deltalr:
        if args.schedule_exp == 0:
            # Step-wise LR decay
            set_lr = args.lr
            for lr_decay in args.schedule:
                if epoch == lr_decay:
                    set_lr *= args.gamma
            state['lr'] = set_lr
            args.lr = set_lr
        else:
            # Exponential LR decay
            set_lr = args.lr
            exp = int((epoch - 1) / args.schedule_exp)
            state['lr'] = set_lr * (args.gamma ** exp)
            args.lr = set_lr
    elif args.deltalr:
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


def visualizePruneTrain(model, epoch, threshold):
    altList = []
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

    if printName:
        print("\naltList", altList)

    printParam = False
    my_cmap = matplotlib.cm.get_cmap('gray', 256)
    newcolors = my_cmap(np.linspace(0, 1, 256))
    pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
    newcolors[:2, :] = pink
    newcmp = ListedColormap(newcolors)
    # print("\ncmap: ", my_cmap(0))
    for a in range(0, len(altList)):
        weight = paramList[a].cpu()
        weight = weight.detach().numpy()

        f_min, f_max = np.min(weight), np.max(weight)
        if printParam:
            print("\nf_min; f_max: ", f_min, " ; ", f_max)
        # When threshold < f_min then no vmin
        weight = (weight - f_min) / (f_max - f_min)

        # threshold = (threshold-f_min)/(f_max-f_min)
        if 'conv' in altList[a]:
            # print("\naltList[", a, "]: ", altList[a])
            dims = paramList[a].shape
            if printParam:
                print("\nParamListShape: ", paramList[a].shape)
            # weight = copy.deepcopy(paramList[a])
            if printParam:
                print("\nDims: ", dims)
            ix = 1
            for i in range(0, dims[0]):  # out channels
                # color = [[[]]]
                ax = None
                filtermap3d = weight[i, :, :, :]
                # print("\nShape FilterMap: ", filtermap3d.shape)
                for j in range(0, dims[1]):  # in channels
                    filterMaps = filtermap3d[j, :, :]

                    if printParam:
                        print("\nWeight: ", filterMaps)

                    ax = pyplot.subplot(dims[0], dims[1], ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pyplot.imshow(filterMaps[:, :], cmap=newcmp)

                    ix += 1
                # printWeights = weightList3d[-j:]
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(printWeights[0], printWeights[1], printWeights[2])
            # pyplot.legend(bbox_to_anchor=(0, -0.15, 1, 0), loc=2, ncol=2, mode="expand", borderaxespad=0)
            fileName = '/img/' + altList[a] + '_' + str(epoch) + '.png'
            pyplot.savefig(fileName)

        elif 'bn' in altList[a]:
            ax = None
            # print("\naltList[", a, "]: ", altList[a])
            dims = paramList[a].shape
            if printParam:
                print("\nParamListShape: ", paramList[a].shape)
            weight = paramList[a].cpu()
            weight = weight.detach().numpy()
            if printParam:
                print("\nDims: ", dims)
            ax = pyplot.plot(weight)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # pyplot.imshow(weight[:,0],cmap=my_cmap,vmin=threshold)
            # ix += 1
            fileName = '/img/' + altList[a] + '_' + str(epoch) + '.png'
            pyplot.savefig(fileName)

        elif 'fc' in altList[a]:
            print("\naltList[", a, "]: ", altList[a])
            dims = paramList[a].shape
            if printParam:
                print("\nParamListShape: ", paramList[a].shape)
            weight = paramList[a].cpu()
            weight = weight.detach().numpy()
            if printParam:
                print("\nDims: ", dims)
            ix = 1
            for i in range(0, dims[0]):  # out channels
                ax = None
            #    if printParam:
            #       print("\nWeight: ", filterMaps)

        #     ax = pyplot.subplot(dims[0], 1, ix)
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        #       pyplot.imshow(weight[i, ], cmap=my_cmap, vmin=threshold, vmax=1)
        #      ix += 1
        #  fileName = altList[a] + '_' + str(epoch) + '.png'
        # pyplot.savefig(fileName)

    pyplot.close('all')


def checkmem(use_gpu):
    total, used, free = \
        os.popen('"nvidia-smi" --query-gpu=memory.total,memory.used,memory.free --format=csv,nounits,noheader'
                 ).read().split('\n')[use_gpu].split(',')
    total = int(total)
    used = int(used)
    free = int(free)
    # print(use_gpu, 'Total GPU mem:', total, 'used:', used)
    return total, used, free


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
