import time
from statistics import mean

import torch
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import model.resnet_cifar10 as resnet
# from pruner.fp_mbnetv2 import FilterPrunerMBNetV2
# from pruner.fp_resnet import FilterPrunerResNet
from pruner.fp_resnet import FilterPrunerResNet
import argparse

from src.utils import Logger

logger = Logger('logMorphNetSize6E6.txt', title='logMorphNet')
logger.set_names(['Regularisierer', 'Zielgroesse', 'Top1'])

def measure_model(model, pruner, img_size):
    pruner.reset() 
    model.eval()
    pruner.forward(torch.zeros((1,3,img_size,img_size), device='cuda'))
    cur_flops = pruner.cur_flops
    cur_size = pruner.cur_size
    return cur_flops, cur_size

def save_checkpoint(state, is_best, filename='checkpoint'):
    if is_best:
        torch.save(state, '{}_best.pth.tar'.format(filename))

def get_valid_flops(model, cbns, out_maps):
    lastConv = None
    residual_chain = {}
    chain_max_dim = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            residual_chain[lastConv] = m.conv[3]
            lastConv = m.conv[3]
            chain_max_dim = np.maximum(chain_max_dim, lastConv.weight.size(1))
        if isinstance(m, nn.Conv2d):
            lastConv = m
            chain_max_dim = lastConv.weight.size(1)

    # Deal with the chain first
    mask = np.zeros(chain_max_dim)
    for key in residual_chain:
        conv = residual_chain[key]
        target_idx = cbns[0].index(conv)
        target_bn = cbns[1][target_idx]
        cur_mask = target_bn.weight.data.cpu().numpy()
        cur_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
        mask = np.logical_or(mask, cur_mask)

    flops = 0
    for idx, (conv, bn) in enumerate(zip(*cbns)):
        if conv in residual_chain:
            cur_mask = mask[:bn.weight.size(0)]
            valid_output = np.sum(cur_mask)
            if idx == 0:
                valid_input = conv.weight.size(1)
            else:
                valid_input = (torch.abs(cbns[1][idx-1].weight) > 0).sum().item()
        else:
            valid_output = (torch.abs(bn.weight) > 0).sum().item()
            cur_mask = mask[:cbns[1][idx-1].weight.size(0)]
            valid_input = np.sum(cur_mask)
        
        flops += out_maps[idx][0] * out_maps[idx][1] * valid_output * valid_input * conv.weight.size(2) * conv.weight.size(3) / conv.groups
    return flops

def get_cbns(model):
    convs = []
    bns = []
    for m in model.modules():
        # store the information for batchnorm
        if isinstance(m, nn.Conv2d):
            convs.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bns.append(m)
    return convs, bns

def regularizer(model, constraint='size', cbns=None, maps=None):
    # build kv map
    if cbns is None:
        cbns = get_cbns(model)
    else:
        G = torch.zeros([1], requires_grad=True).cuda()
        for idx, (conv, bn) in enumerate(zip(*cbns)):
            if idx < len(cbns[0])-1:
                gamma_prev = torch.abs(bn.weight)
                A = (gamma_prev > 0)
                gamma_now = torch.abs(cbns[1][idx+1].weight)
                B = (gamma_now > 0)
                if constraint == 'size':
                    cost = cbns[0][idx+1].weight.size(2)*cbns[0][idx+1].weight.size(3)
                elif constraint == 'flops':
                    assert maps is not None, 'Output Map is None!'
                    cost = 2 * maps[idx+1][0] * maps[idx+1][0] * cbns[0][idx+1].weight.size(2) * cbns[0][idx+1].weight.size(3)
                G = G + cost * (gamma_prev.sum()*B.sum().type_as(gamma_prev) + gamma_now.sum()*A.sum().type_as(gamma_now))
    return G

def num_alive_filters(model):
    cnt = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            cnt = cnt + (torch.abs(m.weight) > 0).sum().item()
    return cnt

# Truncate small beta and enforce depth-wise in-out numbers
def truncate_smallbeta(model, cbns):
    lastConv = None
    residual_chain = {}
    chain_max_dim = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            residual_chain[lastConv] = m.conv[3]
            lastConv = m.conv[3]
            chain_max_dim = np.maximum(chain_max_dim, lastConv.weight.size(1))
        if isinstance(m, nn.Conv2d):
            lastConv = m
            chain_max_dim = np.maximum(chain_max_dim, lastConv.weight.size(1))

    # Deal with the chain first
    mask = np.zeros(chain_max_dim)
    for key in residual_chain:
        conv = residual_chain[key]
        target_idx = cbns[0].index(conv)
        target_bn = cbns[1][target_idx]
        cur_mask = target_bn.weight.data.cpu().numpy()
        zero_idx = np.abs(cur_mask) < 0.01
        cur_mask[zero_idx] = 0
        # print(f'cur mask: {chain_max_dim - len(cur_mask)}')
        cur_mask = np.concatenate((cur_mask, np.zeros(chain_max_dim - len(cur_mask))))
        mask = np.logical_or(mask, cur_mask)

    for idx, (conv, bn) in enumerate(zip(*cbns)):
        weights = bn.weight.data.cpu().numpy()
        bias = bn.bias.data.cpu().numpy()
        if conv in residual_chain:
            cur_mask = mask[:weights.shape[0]]
            weights *= cur_mask
            bias *= cur_mask
        else:
            idx_out = np.abs(weights) < 0.01
            weights[idx_out] = 0
            bias[idx_out] = 0

        bn.weight.data = torch.from_numpy(weights).cuda()
        bn.bias.data = torch.from_numpy(bias).cuda()

def test(model, loader):
    model.eval()
    total = 0
    top1 = 0
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    for i, (batch, label) in enumerate(loader):
        batch, label = batch.to('cuda'), label.to('cuda')
        total += batch.size(0)
        out = model(batch)
        total_loss += criterion(out, label).item()
        _, pred = out.max(dim=1)
        top1 += pred.eq(label).sum()

    return float(top1)/total*100, total_loss/total

def train_epoch(model, optim, criterion, loader, lbda=None, cbns=None, maps=None, constraint=None):
    model.train()
    total = 0
    top1 = 0
    regular =[]
    # print(f'-1')
    for i, (batch, label) in enumerate(loader):
        optim.zero_grad()
        # print(f'1')
        batch, label = batch.to('cuda'), label.to('cuda')
        total += batch.size(0)
        # print(f'2')
        out = model(batch)
        # print(f'3')
        _, pred = out.max(dim=1)
        # print(f'4')
        top1 += pred.eq(label).sum()
        # print(f'5')
        if constraint:
            reg = lbda * regularizer(model, constraint, cbns, maps)
            regN = reg.clone().cpu().detach().numpy()[0]
            # print(f'reg: {regN}')
            regular.append(regN)
            loss = criterion(out, label) + reg
        else:
            loss = criterion(out, label)
        loss.backward()
        optim.step()

        if (i % 100 == 0) or (i == len(loader)-1):
            print('Train | Batch ({}/{}) | Top-1: {:.2f} ({}/{})'.format(
                i+1, len(loader),
                float(top1)/total*100, top1, total))

    meanReg = np.mean(regular)
    print(f'Regular: {meanReg}')
    if constraint:
        truncate_smallbeta(model, cbns)
    return meanReg

def train(model, pruner,  train_loader, val_loader, epochs=10, lr=1e-2, name=''):
    model = model.to('cuda')
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [93,150] , gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    reg =[]
    for e in range(epochs):
        start = time.time()

        regularize = train_epoch(model, optimizer, criterion, train_loader)
        reg.append(regularize)
        flops, num_params = measure_model(pruner.model, pruner, 32)
        ende = time.time() -start

        print(f'Epoche: {e}; regular: {regularize}: flops {flops}')

        top1, _ = test(model, test_loader)
        logger.append([regularize, flops, top1, ende])
        print('#Filters: {}, #FLOPs: {:.2f}M | Top-1: {:.2f}'.format(num_alive_filters(model),
                                                                     pruner.get_valid_flops() / 1000000., top1))

        top1, val_loss = test(model, val_loader)
        print('Epoch {} | Top-1: {:.2f}'.format(e, top1))
        torch.save(model, 'ckpt/{}_best.t7'.format(name))
        # scheduler.step()
    return model

def train_mask(model, train_loader, testloader, pruner, epochs=10, lr=1e-2, lbda=1.3*1e-8, cbns=None, maps=None, constraint='flops'):

    model = model.to('cuda')
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(epochs):
        print('Epoch {}'.format(e))
        start = time.time()
        regularize = train_epoch(model, optimizer, criterion, train_loader, lbda, cbns, maps, constraint)
        flops, num_params = measure_model(pruner.model, pruner, 32)
        print(f'Epoche: {e}; regular: {regularize}: flops {flops}')
        ende = time.time() -start
        top1, _ = test(model, test_loader)
        if constraint == 'flops':
            logger.append([regularize, flops, top1, ende])
        elif constraint == 'size':
            logger.append([regularize, num_params, top1, ende])
        else:
            print(f'FEHLER!!!!!!!!!!!!!!!!!!!!!')
        print('#Filters: {}, #FLOPs: {:.2f}M | Top-1: {:.2f}'.format(num_alive_filters(model), pruner.get_valid_flops()/1000000., top1))
    return model

def prune_model(model, cbns, pruner):
    filters_to_prune_per_layer = pruner.get_valid_filters()
    prune_targets = pruner.pack_pruning_target(filters_to_prune_per_layer, get_segment=True, progressive=True)
    layers_prunned = {}
    for layer_index, filter_index in prune_targets:
        if layer_index not in layers_prunned:
            layers_prunned[layer_index] = 0
        layers_prunned[layer_index] = layers_prunned[layer_index] + (filter_index[1]-filter_index[0]+1)
    print('Layers that will be prunned: {}'.format(sorted(layers_prunned.items())))
    print('Prunning filters..')
    for layer_index, filter_index in prune_targets:
        pruner.prune_conv_layer_segment(layer_index, filter_index)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default='/home/jessica.buehler/MA_Source/dataset/data/torch')
    parser.add_argument("--logger", type=str, default='/home/jessica.buehler/MA_Source/MorphLogs/log.txt')
    parser.add_argument("--dataset", type=str, default='torchvision.datasets.CIFAR10')
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--inPlanes", type=int, default=16)
    parser.add_argument("--name", type=str, default='ft_mbnetv2')
    parser.add_argument("--model", type=str, default='ft_mbnetv2')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--lbda", type=float, default=3e-8)
    parser.add_argument("--prune_away", type=float, default=1, help='The constraint level in portion to the original network, e.g. 0.5 is prune away 50%')
    parser.add_argument("--constraint", type=str, default='flops')
    parser.add_argument("--large_input", action='store_true', default=False)
    parser.add_argument("--no_grow", action='store_true', default=False)
    parser.add_argument("--pruner", type=str, default='FilterPrunnerResNet', help='Different network require differnt pruner implementation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(args)
    model = resnet.ResNet32(num_classes=10, inPlanes = args.inPlanes)
    model.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_set = eval(args.dataset)(args.datapath, True, transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_set = eval(args.dataset)(args.datapath, True, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    num_train = len(train_set)
    indices = list(range(num_train))
    # split = int(np.floor(0.1 * num_train))

    np.random.seed(98)
    np.random.shuffle(indices)

    # train_idx, valid_idx = indices[split:], indices[:split]
    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    test_set = eval(args.dataset)(args.datapath, False, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=6, pin_memory=True
    )
    # val_loader = torch.utils.data.DataLoader(
    #     val_set, batch_size=args.batch_size, sampler=valid_sampler,
    #     num_workers=6, pin_memory=True
    # )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=125, shuffle=False,
        num_workers=6, pin_memory=False
    )
    logger = Logger(args.logger, title='logMorphNet')
    logger.set_names(['Regularisierer', 'Zielgroesse', 'Top1', 'Trainingszeit'])
    if 'CIFAR10' in args.dataset:
        train_set.num_classes = 10
    elif 'CIFAR100' in args.dataset:
        train_set.num_classes = 100
    pruner = FilterPrunerResNet(model, 'l2_weight', num_cls=train_set.num_classes)
    flops, num_params = measure_model(pruner.model, pruner, 32)

    target = int(args.prune_away * flops)
    for i in range(0,1):
        print(f'I: {i}')
        print(f'flops: {flops}')
        maps = pruner.omap_size
        cbns = get_cbns(pruner.model)
        print('Before Pruning | FLOPs: {:.3f}M | #Params: {:.3f}M'.format(flops / 1000000., num_params / 1000000.))
        train_mask(pruner.model, train_loader, test_loader, pruner, epochs=args.epoch, lr=args.lr, lbda=args.lbda, cbns=cbns,
                   maps=maps, constraint=args.constraint)
        prune_model(pruner.model, cbns, pruner)
        print('Target ({}): {:.3f}M'.format(args.constraint, target / 1000000.))
        flops, num_params = measure_model(pruner.model, pruner, 32)
        # logger.append([0, num_params, 0, 0])

        print('After Pruning | FLOPs: {:.3f}M | #Params: {:.3f}M'.format(flops / 1000000., num_params / 1000000.))
        if args.no_grow:
            i = 1
        else:
            if flops < target:
                ratio = pruner.get_uniform_ratio(target)
                print(ratio)
                pruner.uniform_grow(ratio)
                flops, num_params = measure_model(pruner.model, pruner, 32)
                print('After Growth | FLOPs: {:.3f}M | #Params: {:.3f}M'.format(flops/1000000., num_params/1000000.))
                # train(pruner.model, train_loader, test_loader, epochs=args.epoch, lr=args.lr, name=args.name)

    #train_mask(pruner.model, train_loader, test_loader, pruner, epochs=30, lr=args.lr, lbda = args.lbda,
    #           cbns = cbns, maps = maps, constraint=None)
    print(f'model: {model}')
    test_acc, test_loss = test(pruner.model,test_loader)
    print(f'Test acc: {test_acc}')
    # logger.append([0,0,test_acc])