import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from torch.autograd import Variable

import n2n

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
parser.add_argument('--gpu_id', default='2', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

best_acc = 0

def main():
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    if args.manualSeed is None:
        args.manualSeed = torch.random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed(args.manualSeed)

    torch.autograd.set_detect_anomaly(True)

    if use_cuda:
        torch.cuda.manual_seed(args.manualSeed)

    torch.autograd.set_detect_anomaly(True)
    global best_acc
    # Data
    print('==> Preparing dataset %s' % args.dataset)
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):
        train(trainloader, model, criterion, optimizer, epoch, use_cuda)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    print("Train")
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # inputs = Variable(inputs)
        # target = torch.autograd.Variable(targets)

        with torch.no_grad():
            inputs = Variable(inputs)
        targets = torch.autograd.Variable(targets)
        outputs = model.forward(inputs)

        loss = criterion(outputs, targets)

        # lasso penalty
        init_batch = batch_idx == 0 and epoch == 1

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, param in model.named_parameters():
            # Get Momentum parameters to adjust
            print(name)
            mom_param = optimizer.state[param]['momentum_buffer']
