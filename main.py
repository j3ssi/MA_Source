import copy
import os
import argparse
import shutil
import time
import random

from torchviz import make_dot
import matplotlib
import numpy as np
from torch.nn.utils import clip_grad_norm_
from src.lars import LARS

from src.mem_reporter import *
from copy import deepcopy

from torch.optim.lr_scheduler import StepLR

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
from torchtest import assert_vars_change

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
parser.add_argument('-n', type=str, help="#blocks per stag numbers separated by commas")
parser.add_argument('-b', '--bottleneck', default=False, action='store_true',
                    help='Set the bootleneck parameter')
parser.add_argument('-w', '--widthofFirstLayer', default=16, type=int,
                    help='defines the width of the first stage in net')

# epochs and stuff
parser.add_argument('--epochs', default=8, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-r', '--reset', default=False, action='store_true',
                    help='Last Epoch')
parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochsFromBegin', default=1, type=int, metavar='N',
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
parser.add_argument('-dB', default=False, action='store_true',
                    help='Use large or small batch size')

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
parser.add_argument('--manualSeed', type=int, default=6, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--dynlr', default=False, action='store_true',
                    help='Dynamische LR')
parser.add_argument('--schedule-exp', type=int, default=0, help='Exponential LR decay.')

# PruneTrain
parser.add_argument('--sparse_interval', default=0, type=int,
                    help='Interval to force the value under threshold')
parser.add_argument('--threshold', default=0.0001, type=float,
                    help='Threshold to force weight to zero')
parser.add_argument('--en_group_lasso', default=False, action='store_true',
                    help='Set the group-lasso coefficient')
parser.add_argument('--scheduler', default=False, action='store_true',
                    help='Use scheduler')

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
parser.add_argument('--n2n', default=False, action='store_true',
                    help='Use net2net functionality')
parser.add_argument('--wider', default=0, type=int,
                    help='Make network wider')
parser.add_argument('--deeper', default=0, type=int,
                    help='Make network deeper')
parser.add_argument('--deeper2', default=0, type=int,
                    help='Make network deeper')

parser.add_argument('--widerRnd', default=0, type= int,
                    help='Make network wider')
parser.add_argument('--widthOfAllLayers', type=str, help="#width of stages separated by commas")

# lars
parser.add_argument('--lars', default=False, action='store_true',
                    help='use lars')
parser.add_argument('--larsLR', default=0.001, type=float,
                    help='')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

listofBlocks = [int(i) for i in args.n.split(',')]
print(listofBlocks)
listOfWidths = []
if args.widthOfAllLayers is not None:
    listOfWidths = [int(i) for i in args.widthOfAllLayers.split(',')]
    print(listOfWidths)

print(
    f'Pytorch Training main.py; workers: {args.workers}; numOfStages: {args.numOfStages}; layerinBlock: {args.layersInBlock};'
    f'widthofFirstLayer: {args.widthofFirstLayer}; Epochen: {args.epochs}; reset: {args.reset}; start epoche: {args.start_epoch}; test: {args.test} '
    f'pathtoModell: {args.pathToModell}; checkpoint: {args.checkpoint}; saveModell: {args.saveModell}; LR: {args.lr}')


def main():
    # checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    use_cuda = torch.cuda.is_available()

    # Random seed
    args.manualSeed = random.randint(1, 10000)
    print(f'random number: {args.manualSeed}')
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

    title = 'prune' + str(args.epochsFromBegin)
    if args.resume:
        model = torch.load(args.pathToModell)
        model.cuda()
        # print(f'Model: {model}')
        criterion = nn.CrossEntropyLoss()
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        if args.dB:
            memory = checkpoint['memory']
            batch_size = checkpoint['batch_size']
        start_epoch = checkpoint['epoch']
        print(f'Start epoch: {start_epoch}')
        optimizer = checkpoint['optimizer']
        print(f'First Lr: {optimizer.param_groups[0]["lr"]}')
        # if args.scheduler:
        #     if checkpoint['optimizer'] is not None:
        #         scheduler = checkpoint['scheduler']
        #     else:
        #         scheduler = StepLR(optimizer, step_size=30, gamma=0.95)        # start_batchSize = checkpoint['start_batchSize']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        logger.set_names(
            ['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc.', 'ValidAcc.', 'TrainEpochTime(s)',
             'TestEpochTime(s)'])


    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(
            ['LearningRate', 'TrainLoss', 'ValidLoss', 'TrainAcc.', 'ValidAcc.' ,'TrainEpochTime(s)',
             'TestEpochTime(s)'])
        assert args.numOfStages == len(
            listofBlocks), 'Liste der Blöcke pro Stage sollte genauso lang sein wie Stages vorkommen!!!'
        memory = 0
        if len(listOfWidths) > 0:
            model = n2n.N2N(num_classes, args.numOfStages, listofBlocks, args.layersInBlock, True,
                            widthofFirstLayer=16, model=None, archNums=None, widthOfLayers=listOfWidths)
        else:
            model = n2n.N2N(num_classes, args.numOfStages, listofBlocks, args.layersInBlock, True,
                            widthofFirstLayer=args.widthofFirstLayer, model=None, archNums=None, widthOfLayers=None)

        print(f'device count: {torch.cuda.device_count()}')
        # model.cuda()
        criterion = nn.CrossEntropyLoss()
        start_epoch = 1
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print(f'Startepoche: {start_epoch}')
    # print(f'Max memory: {torch.cuda.max_memory_allocated() / 10000000}')

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Count the parameters of the model and calculate training bacth size
    count0 = 0
    for p in model.parameters():
        count0 += p.data.nelement()
    # print(f'count0: {count0}')
    count1 = count0

    # Calculate Size of Trainings Batch size
    if not args.batchTrue and args.epochsFromBegin == 0:

        # calculate first how many blocks is equal to the count0
        sizeX = (count0 - 1306) / 97216
        # Gerade für niedrige Batch size
        # if not args.largeBatch:
        y = 68.25 * sizeX + 47.85
        # else:
        # y = 4.27 * sizeX + 2.60
        # calculate now the batch size
        batch_size = int(0.999 * count0 / sizeX / y)
        # delta_bs = (batch_size - 330)*0.3
        # batch_size = int(batch_size - delta_bs)
        print(f'batch_size berechnet: {batch_size};{y} ; lr: {args.lr}')
        args.batch_size = batch_size
    elif args.batchTrue:
        batch_size = args.batch_size
    batch_size = int(batch_size)
    # args.lr *= (batch_size / 256)

    if not args.resume:
        start_batchSize = batch_size

    trainloader = data.DataLoader(trainset, batch_size=batch_size, pin_memory=True,
                                  shuffle=True, num_workers=args.workers)
    #    optimizer = LARS(model.parameters(), eta=args.larsLR, lr=args.lr, momentum=args.momentum,
    #                     weight_decay=args.weight_decay)
    k = args.epochs/6
    n1 = 30
    n2 = 3
    i = 1
    # for epochNet2Net in range(1, 4):
    print(f'deeper epoch: {args.deeper}')
    testacc = []
    wAvg = []
    while i == 1:
        for epoch in range(start_epoch, args.epochs + start_epoch):

            if args.delta_learning_rate:
                optimizer.param_groups[0]["lr"] = adjust_learning_rate(optimizer, epoch, True)

            print(f'Epoche: [{epoch}/{args.epochs + start_epoch - 1}]; Lr: {optimizer.param_groups[0]["lr"]}')
            print(f'batch Size {batch_size}')
            # start = time.time()
            torch.cuda.reset_max_memory_allocated()
            print(f'befor train')
            train_loss, train_acc, train_epoch_time = train(trainloader, model, criterion,
                                                            optimizer, epoch, use_cuda)
            # ende = time.time()
            tmp_memory = torch.cuda.max_memory_allocated()

            # print(f'lr: {optimizer.param_groups[0]["lr"]}')
            # if args.dynlr and scheduler is not None:
            #     # adjust_learning_rate(optimizer, epoch, False)
            #     scheduler.step()
            #     lr = scheduler.get_last_lr()[0]
            #     print(f'args.lr: {lr}')
            # print(f'lr: {optimizer.param_groups[0]["lr"]}')

            # print(f'Max memory in training epoch: {torch.cuda.max_memory_allocated() / 10000000}')
            test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, epoch, use_cuda)
            testacc.append(test_acc)
            y = []
            n1 = 30
            if n1 > epoch:
                n1 = epoch
            y.append( testacc[-1] )
            a1 = 2 / (n1 + 1)
            #
            # try:
            #     wAcc = testacc[ - n1 ]
            # except:
            #     wAcc = 0
            # print(f'n1: {n1} for:')
            # k = n1
            # while k > 1:
            #     wAcc = (1-a1) * wAcc + a1 *testacc[ - k ]
            #     # print(f'n1: {k}')
            #     k = k - 1
            # # if epoch > 1:

            # y.append(wAcc)
            # print(f'wAcc: {wAcc}')
            print(f'test acc: {test_acc}')

            # print(f'n1: {n1}')
            # append logger file
            logger.append(
               [optimizer.param_groups[0]["lr"], train_loss, test_loss, train_acc, test_acc, train_epoch_time,
                test_epoch_time])
            countB = 0
            for p in model.parameters():
                countB += p.data.nelement()

            # i = 2
            # SparseTrain routine
            if not args.en_group_lasso:
                pass
            elif args.en_group_lasso and (epoch % args.sparse_interval == 0) and not args.lastEpoch:
                # Force weights under threshold to zero
                print(f'Prune Train:')
                dense_chs, chs_map = makeSparse(optimizer, model, args.threshold)
                if args.visual:
                    visualizePruneTrain(model, epoch, args.threshold)
                print(f'Dense channels: {dense_chs}')
                genDenseModel(model, dense_chs, optimizer)
                model.newModuleList(num_classes)
                gc.collect()
                model.cuda()
                if not args.lars:
                    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
                else:
                    optimizer = LARS(model.parameters(), eta=args.larsLR, lr=args.lr, momentum=args.momentum)  # ,
                    # weight_decay=args.weight_decay)

            if args.visual:
                visualizePruneTrain(model, epoch, args.threshold)

            # if args.fp16:
            #   model, optimizer = amp.initialize(model, optimizer)
            #
            count = 0
            # print(f'count0: {count0}')
            for p in model.parameters():
                count += p.data.nelement()
            if count < count1:
                print(f'Count: {count} ; {count0} ; {count / count0}')
                count1 = count

            if args.dB and (epoch % 5) == 3:
                print(f'Drin!!')
                print(f'old memory: {memory}')
                print(f'new memory: {tmp_memory}')
                if memory > 0:
                    factor = tmp_memory / memory
                    print(f'Faktor: {factor}')

                    if factor < 1:
                        batch_size_tmp = int(669950000 / tmp_memory * batch_size)
                        batch_size = batch_size_tmp
                        memory = tmp_memory
                        print(f'New batch Size größer {batch_size}!!')
                    if factor > 1:
                        batch_size_tmp = int(tmp_memory / memory * batch_size)
                        batch_size = batch_size_tmp
                        memory = tmp_memory
                        print(f'New batch Size kleiner {batch_size}!!')

            elif args.dB and epoch % 5 == 4:
                memory = tmp_memory
            if args.deeper == epoch:
                print("\n\nnow deeper1")
                # deeper student training
                model.deeper(pos=1)
                # batch_size = 512
                print(f'Nums: {model.archNums}')
                print(
                    f'num: {num_classes}; numofstages: {args.numOfStages}, listofBlocks: {listofBlocks}, layers in blocj: {args.layersInBlock}')
                # model.newModuleList(num_classes)
                model.cuda()
                criterion = nn.CrossEntropyLoss()
                # optimizer = optim.Adam(model.parameters())
                print(f'model.para: {model.named_parameters()}')
                # optimizer = LARS(model.parameters(), eta=args.larsLR, lr=args.lr, momentum=args.momentum,
                #                 weight_decay=args.weight_decay)
                # state['lr'] = 0.001
                # args.lr = 0.001

                optimizer = optim.SGD(model.parameters(), lr=args.lr,
                                      momentum=args.momentum)  # , weight_decay=args.weight_decay)

                # scheduler = StepLR(optimizer, step_size=30, gamma=0.95)
                # print(model)
            if args.wider == epoch:
                start = time.time()
                model.wider(2, weight_norm=None, random_init=False, addNoise=True)
                # model.widthofLayers[0] *= 2
                for i in range(len(model.widthofLayers)):
                   model.widthofLayers[i] *= 2
                model.newModuleList(10)
                model.cuda()
                # print(model)
                # criterion = nn.CrossEntropyLoss()
                ende = time.time() -start
                print(f'time for n2n: {ende}')
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            if args.widerRnd == epoch:
                model.wider(2, weight_norm=None, random_init=True, addNoise=True)
                # model.widthofLayers[0] *= 2
                for i in range(len(model.widthofLayers)):
                   model.widthofLayers[i] *= 2
                model.newModuleList(10)
                model.cuda()
                # print(model)
                # criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            if args.deeper2 == epoch:
                print("\n\nnow deeper1")
                # deeper student training
                model.deeper2(pos = 1)
                print(f'args.layersInBlock: {args.layersInBlock}')
                model.newModuleList(num_classes= 10)
                model.cuda()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
                print(model)

            # for param in model.paramList:
            #    print(f'Parameter: {param.data}')
            # for param in model.paramList1:
            #    print(f'Parameter1: {param.data}')
        i = 2

    # print(f'model parameters: {list(model.named_parameters())}')

    # scheduler = StepLR(optimizer, step_size=60, gamma=0.75)
    # if args.widerRnd and not args.wider:
    #     model = model.wider(3, 2, out_size=None, weight_norm=None, random_init=True, addNoise=False)
    #
    #     model = model.wider(2, 2, out_size=None, weight_norm=None, random_init=True, addNoise=False)
    #
    #     model = model.wider(1, 2, out_size=None, weight_norm=None, random_init=True, addNoise=False)
    #
    #     model = n2n.N2N(num_classes, args.numOfStages, listofBlocks, args.layersInBlock, False, args.bottleneck,
    #                     widthofFirstLayer=16, model=model, archNums=model.archNums, widthOfLayers=listOfWidths)
    #     print(model)
    #     model.cuda()
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.SGD(model.parameters(), lr=optimizer.param_groups[0]["lr"], momentum=args.momentum,
    #                           weight_decay=args.weight_decay)
    #     scheduler = StepLR(optimizer, step_size=60, gamma=0.75)

    # print("Test acc1: ", test_acc)
    # test_loss, test_acc, test_epoch_time = test(testloader, model, criterion, 1, use_cuda)
    # print("Test acc2: ", test_acc)
    if not args.scheduler:
        scheduler = None

    print("[INFO] Storing checkpoint...")
    if args.reset:
        print(f'Reset! ')
        start_epoch = 1
        args.epochs = 0

        save_checkpoint({
            'epoch': args.epochs + start_epoch,
            'lr': optimizer.param_groups[0]["lr"],
            'acc': test_acc,
            'optimizer': optimizer,
            'scheduler': scheduler},
            checkpoint=args.checkpoint)

    if args.dB:
        save_checkpoint({
            'epoch': start_epoch + args.epochs,
            'memory': memory,
            'batch_size': batch_size,
            'lr': optimizer.param_groups[0]["lr"],
            'acc': test_acc,
            'scheduler': scheduler,
            'optimizer': optimizer, },
            checkpoint=args.checkpoint)
    else:
        save_checkpoint({
            'epoch': args.epochs + start_epoch,
            'lr': optimizer.param_groups[0]["lr"],
            'acc': test_acc,
            'scheduler': scheduler,
            'optimizer': optimizer, },
            checkpoint=args.checkpoint)

    # Leave unique checkpoint of pruned models druing training
    if epoch % args.save_checkpoint == 0:
        if args.dB:
            save_checkpoint({
                'epoch': args.epochs + start_epoch,
                'memory': memory,
                'batch_size': batch_size,
                'scheduler': scheduler,
                'lr': optimizer.param_groups[0]["lr"],
                'acc': test_acc,
                'optimizer': optimizer, },
                checkpoint=args.checkpoint)
        else:
            save_checkpoint({
                'epoch': args.epochs + start_epoch,
                'acc': test_acc,
                'scheduler': scheduler,
                'optimizer': optimizer, },
                checkpoint=args.checkpoint,
                filename='checkpoint' + str(epoch) + '.tar')

    if args.saveModell:
        torch.save(model, args.pathToModell)
    logger.close()

    print(f'Max memory: {torch.cuda.max_memory_allocated() / 10000000}')
#    print(' {:5.3f}s'.format(ende - start), end='  ')


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
    printLasso = False

    lasso_ratio = AverageMeter()
    # print(f'test 4')
    end = time.time()
    # print(f'test 3')
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(f'test 2')
        # measure data loading time
        data_time.update(time.time() - end)
        data_load_time = time.time() - end

        if use_cuda:
            # print(f'test -2')
            inputs, targets = inputs.cuda(), targets.cuda()
        # print(f'test -1')
        with torch.no_grad():
            inputs = Variable(inputs)
        targets = torch.autograd.Variable(targets)
        # print(f'Test')
        outputs = model(inputs)
        # print(f'Test2')
        loss = criterion(outputs, targets)
        # if batch_idx == 0 and (epoch == 10):
        #     dot = tw.make_dot(outputs, params=dict(model.named_parameters()))
        #     if len(model.module_list) < 60:
        #         filename = 'model/n2nBefore' + str(epoch) + '_' + str(batch_idx) + '.dot'
        #     else:
        #         filename = 'model/n2nAfter' + str(epoch) + '_' + str(batch_idx) + '.dot'
        #
        #     dot.render(filename=filename)

        # lasso penalty
        init_batch = False # batch_idx == 0 and epoch == 1

        # if args.en_group_lasso:
            # if args.global_group_lasso:
            #     lasso_penalty = get_group_lasso_global(model)
            # else:
            #    lasso_penalty = get_group_lasso_group(model)
        #     if printLasso:
        #         print(f'Lasso Penalty1: {lasso_penalty}')
        #         # Auto-tune the group-lasso coefficient @first training iteration
        #         # coeff_dir = os.path.join(args.coeff_container, 'cifar')
        #     if init_batch:
        #         args.grp_lasso_coeff = args.var_group_lasso_coeff * loss.item() / (
        #                 lasso_penalty * (1 - args.var_group_lasso_coeff))
        #         grp_lasso_coeff = torch.autograd.Variable(args.grp_lasso_coeff)
        #
        #         with open(os.path.join(args.checkpoint, str(args.var_group_lasso_coeff)), 'w') as f_coeff:
        #             f_coeff.write(str(grp_lasso_coeff.item()))
        #         if printLasso:
        #             print(f'Grp lasso coeff: {grp_lasso_coeff}')
        #
        #     else:
        #         with open(os.path.join(args.checkpoint, str(args.var_group_lasso_coeff)), 'r') as f_coeff:
        #             for line in f_coeff:
        #                 grp_lasso_coeff = float(line)
        #     lasso_penalty = lasso_penalty * grp_lasso_coeff
        #     if printLasso:
        #         print(f'Lasso Penalty2: {lasso_penalty}')
        # else:
        #     lasso_penalty = 0
        #     # print(f'nach group lasso')
        #     # Group lasso calcution is not performance-optimized => Ignore from execution time
        # loss += lasso_penalty
        # # measure accuracy and record loss

        prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # lasso_ratio.update(lasso_penalty / loss.item(), inputs.size(0))

        optimizer.zero_grad()
        #   compute gradient and do SGD step
        # if args.O1 or args.O2 or args.O3:
        #     with amp.scale_loss(loss, optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        loss.backward()
        # plot_grad_flow(model.named_parameters(), epoch)
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
        # break


    print(f'after train')
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


def adjust_learning_rate(optimizer, epoch, change_lr):
    # Step-wise LR decay
    lr = optimizer.param_groups[0]["lr"]
    for lr_decay in args.schedule:
        if epoch == lr_decay:
            lr *= args.gamma
    state['lr'] = lr
    args.lr = lr
    #     else:
    #         print(f'2')
    #         # Exponential LR decay
    #         lr = optimizer.param_groups[0]["lr"]
    #         exp = int((epoch - 1) / args.schedule_exp)
    #         lr *= (args.gamma ** exp)
    # else:
    #     if args.schedule_exp == 0:
    #         print(f'3')
    #         # Step-wise LR decay
    #         lr = optimizer.param_groups[0]["lr"]
    #         for lr_decay in args.schedule:
    #             if epoch >= lr_decay:
    #                 lr *= args.gamma
    #     else:
    #         print(f'4')
    #         # Exponential LR decay
    #         lr = optimizer.param_groups[0]["lr"]
    #         exp = int((epoch - 1) / args.schedule_exp)
    #         lr *= (args.gamma ** exp)
    return lr


def plot_grad_flow(named_parameters, epoche):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    fileName = 'gradflow' + str(epoche) + '.png'
    plt.savefig(fileName)


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
    blue = np.array([70 / 256, 130 / 256, 180 / 256, 1])
    newcolors[:2, :] = blue
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
        if 'conv1' in altList[a]:
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
            fileName = altList[a] + '_' + str(epoch) + '.png'
            pyplot.savefig(fileName)

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


def save_checkpoint(state, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)


if __name__ == '__main__':
    main()
