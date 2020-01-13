import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import optim

from torch.autograd import Variable


class N2N(nn.Module):

    def __init__(self, num_classes):
        super(N2N, self).__init__()
        self.module_list = nn.ModuleList()
        # 0
        conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv1)
        # 1
        bn1 = nn.BatchNorm2d(16)
        self.module_list.append(bn1)

        # 2
        conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv2)
        # 3
        bn2 = nn.BatchNorm2d(16)
        self.module_list.append(bn2)
        # 4
        conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv3)
        # 5
        bn3 = nn.BatchNorm2d(16)
        self.module_list.append(bn3)

        # 6
        conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv4)
        # 7
        bn4 = nn.BatchNorm2d(16)
        self.module_list.append(bn4)
        # 8
        conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv5)
        # 9
        bn5 = nn.BatchNorm2d(16)
        self.module_list.append(bn5)

        # 10
        conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv6)
        # 11
        bn6 = nn.BatchNorm2d(16)
        self.module_list.append(bn6)
        # 12
        conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv7)
        # 13
        bn7 = nn.BatchNorm2d(16)
        self.module_list.append(bn7)

        # 14
        conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv8)
        # 15
        bn8 = nn.BatchNorm2d(16)
        self.module_list.append(bn8)
        # 16
        conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv9)
        # 17
        bn9 = nn.BatchNorm2d(16)
        self.module_list.append(bn9)

        # 18
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.module_list.append(avgpool)
        # 19
        fc = nn.Linear(16, num_classes)
        self.module_list.append(fc)
        self.relu = nn.ReLU(inplace=True)

        for m in self.module_list:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        odd = False
        first = True
        bn = False
        _x = None
        # i=0
        for module in self.module_list:
            if isinstance(module, nn.AdaptiveAvgPool2d):
                try:
                    x = module(_x)
                    x = x.view(-1, 16)

                # print("\navgpool", i)
                # i = i + 1
                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print("AvgPool")
            elif isinstance(module, nn.Linear):
                x = module(x)
                # print("\nfc", i)
                return x
            else:
                if first and not bn:
                    x = module(x)
                    bn = True
                #          print("\nFirst conv", i)
                #         i = i+1
                elif first and bn:
                    x = module(x)
                    _x = self.relu(x)
                    #        print("\nFirst bn", i)
                    #       i = i+1
                    first = False
                    bn = False
                else:
                    if not odd and not bn:
                        x = module(_x)
                        #          print('\nconv',i)
                        #         i=i+1
                        bn = True
                    elif not odd and bn:
                        x = module(x)
                        x = self.relu(x)
                        #        print("\nbn",i)
                        #       i=i+1
                        odd = True
                        bn = False
                    else:
                        if not bn:
                            x = module(x)
                            bn = True
                        #          print('Odd conv',i)
                        #         i=i+1
                        elif bn:
                            x = module(x)
                            _x = _x + x
                            _x = self.relu(_x)
                            odd = False
                            bn = False

    def deeper(self, model, optimizer, positions):
        # each pos in pisitions is the position in which the layer sholud be duplicated to make the cnn deeper
        for pos in positions:
            # print("\n\nposition:")
            # print(pos)
            conv = model.module_list[pos * 2 - 2]
            bn = model.module_list[pos * 2 - 1]
            conv1 = model.module_list[pos * 2]
            bn1 = model.module_list[pos * 2 + 1]
            conv2 = copy.deepcopy(conv)
            conv3 = copy.deepcopy(conv1)
            noise = torch.Tensor(conv2.weight.shape).random_(0, 1).cuda()
            # noise = torch.rand(0,0.5)
            conv2.weight.data += noise
            bn2 = copy.deepcopy(bn)
            noise = torch.Tensor(conv1.weight.shape).random_(0, 1).cuda()
            conv3.weight.data += noise
            bn3 = copy.deepcopy(bn1)
            model.module_list.insert(pos * 2 + 2, conv2)
            model.module_list.insert(pos * 2 + 3, bn2)
            model.module_list.insert(pos * 2 + 4, conv3)
            model.module_list.insert(pos * 2 + 5, bn3)
            # print("\n\n> moduleList:\n")
            # print(self.module_list)

        # optimizer = optim.SGD(model.parameters(), get_lr(optimizer), get_momentum(optimizer),
        #                      get_weight_decay(optimizer))

        return model, optimizer


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def getResidualPath(model):
    stages = {0: {}}

    stages[0]['i'] = []
    stages[0]['o'] = []
    i = int((len(model.module_list) - 2) / 2 + 1)
    listI = []
    listO = []
    for j in range(1, i):
        if j % 2 == 0:
            listI.append(n(j))
        else:
            listO.append(n(j))
    stages[0]['o'] = listO
    stages[0]['i'] = listI
    print(stages)
    return stages


def getShareSameNodeLayers(model):
    sameNode = []
    i = int((len(model.module_list) - 2) / 2)
    for j in range(2, i):
        if j % 2 == 0:
            sameNode.append((n(j), n(j + 1)))
    return sameNode


def n(name):
    if isinstance(name, int):
        return 'module.conv' + str(name) + '.weight'
    else:
        return 'module.' + name + '.weight'



def getRmLayers(name, dataset):
    pass
