import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import n2n


def main():
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed(args.manualSeed)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc, lasso_ratio, train_epoch_time = train(trainloader, model, criterion, optimizer,
                                                                 epoch, use_cuda)
def train(trainloader, model, criterion, optimizer, epoch, use_cuda):

    model.train()

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
