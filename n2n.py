import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable


class N2N(nn.Module):

    def __init__(self, num_classes):
        super(N2N, self).__init__()
        self.module_list = nn.ModuleList()
        #0
        conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv1)
        #1
        bn1 = nn.BatchNorm2d(16)
        self.module_list.append(bn1)
        #2
        conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv2)

        #3
        bn2 = nn.BatchNorm2d(16)
        self.module_list.append(bn2)
        #4
        conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv3)
        #5
        bn3 = nn.BatchNorm2d(16)
        self.module_list.append(bn3)
        #6
        conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv4)
        #7
        bn4 = nn.BatchNorm2d(16)
        self.module_list.append(bn4)
        #8
        conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv5)
        #9
        bn5 = nn.BatchNorm2d(16)
        self.module_list.append(bn5)
        #10
        conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv6)
        #11
        bn6 = nn.BatchNorm2d(16)
        self.module_list.append(bn6)
        #12
        conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv7)
        #13
        bn7 = nn.BatchNorm2d(16)
        self.module_list.append(bn7)
        # 14
        conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv8)
        #15
        bn8 = nn.BatchNorm2d(16)
        self.module_list.append(bn8)
        #16
        conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv9)
        #17
        bn9 = nn.BatchNorm2d(16)
        self.module_list.append(bn9)
        # 18
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.module_list.append(avgpool)
        #19
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
        print("\n\n> moduleList:\n")
        print(self.module_list)

    def forward(self, x):

        x = self.module_list[0](x)
        x = self.module_list[1](x)
        _x = self.relu(x)
        i = 2
        while i > 0:
            if isinstance(self.module_list[i],nn.AdaptiveAvgPool2d):
                try:
                    x = self.module_list[i](_x)
                    x = x.view(-1, 16)
                    x = self.module_list[i+1](x)
                    return x
                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print(i)

            if isinstance(self.module_list[i], nn.Conv2d):
                try:
                    x = self.module_list[i](_x)
                    i = i+1
                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print(i)

            if isinstance(self.module_list[i], nn.BatchNorm2d):
                try:
                    x = self.module_list[i](x)
                    i = i+1

                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print(i)

            x = self.relu(x)

            if isinstance(self.module_list[i], nn.Conv2d):
                try:
                    x = self.module_list[i](x)
                    i = i + 1

                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print(i)

            if isinstance(self.module_list[i], nn.BatchNorm2d):
                try:
                    x = self.module_list[i](x)
                    i = i+1
                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print(i)
            _x = _x + x
            _x = self.relu(_x)

    def deeper(self, model, positions):
        # each pos in pisitions is the position in which the layer sholud be duplicated to make the cnn deeper
        for pos in positions:
            print("\n\nposition:")
            print(pos)
            conv = model.module_list[pos*2-2]
            bn = model.module_list[pos*2-1]
            conv2 = copy.deepcopy(conv)
            noise = torch.Tensor(conv2.shape())
            noise = torch.rand(0,0.5)
            conv2.weight.data += noise
            bn2 = copy.deepcopy(bn)
            model.module_list.insert(pos*2+2, conv2)
            model.module_list.insert(pos*2+3, bn2)
        return model

        #     if posStr in name:
        #         i = name.index(posStr)
        #         conv1 = module[i]
        #         conv2 = conv1.clone()
        #         convStr2 = 'conv' + str(j)
        #         if convStr2 not in names:
        #             posStr = 'conv' + str(j)
        #             self.add_module()
        #             print(self.__dict__.__getitem__('_modules'))
        #             return model
        #         else:
        #             conv3 = module[i + 1]
        #             posStr = 'conv' + str(pos + 1)
        #             j = j + 1
        # for name, module in model.named_parameters():
        #     posStr = 'conv' + str(pos + i)
        #     #    posStr1 = 'conv' + posModel
        #     #   name[posModel] = posStr1
        #     #  model[posModel + 1] = model[posModel]
        #     # model[posModel] = conv2
        #     # else:
        #     #   print(name[posModel])


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features
