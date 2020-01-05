import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.autograd import Variable


class N2N(nn.Module):

    def __init__(self, num_classes, is_deeper=False):
        super(N2N, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        if not is_deeper:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn1 = nn.BatchNorm2d(16)

            # 1
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn3 = nn.BatchNorm2d(16)

            # 2
            self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn4 = nn.BatchNorm2d(16)
            self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn5 = nn.BatchNorm2d(16)

            # 3
            self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn6 = nn.BatchNorm2d(16)
            self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn7 = nn.BatchNorm2d(16)

            # 4
            self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn8 = nn.BatchNorm2d(16)
            self.conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.bn9 = nn.BatchNorm2d(16)

            # 5
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(16, num_classes)
            self.relu = nn.ReLU(inplace=True)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        _x = self.relu(x)
        i = 2
        while i > 0:
            convStr = 'conv' + str(i)
            names = self.__dict__.__getitem__('_modules')

            if convStr not in names:
                # print("\nX.size:\n")
                # print(_x.size(1))
                # print("\n")
                # print(_x.size(2))
                # print("\n")
                # print(_x.size(3))
                # print("\n")

                x = self.avgpool(_x)

                # print("\nX.size:\n")
                # print(x.size(1))
                # print("\n")
                # print(x.size(2))
                # print("\n")
                # print(x.size(3))
                # print("\n")

                #
                #
                x = x.view(-1, 16)
                # #x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
            # find the module with name convStr
            for name, module in self.named_modules():
                if name == convStr:
                    try:
                        x = module.forward(_x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")

            bnStr = 'bn' + str(i)
            for name, module in self.named_modules():
                if name == bnStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")
            x = self.relu(x)
            i = i + 1
            convStr = 'conv' + str(i)
            for name, module in self.named_modules():
                if name == convStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")

            bnStr = 'bn' + str(i)

            for name, module in self.named_modules():
                if name == bnStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")
            try:
                _x = _x + x
            except RuntimeError:
                print("\n \n Oops!!  \n \n \n")

            _x = self.relu(_x)
            i = i + 1

    def deeper(self, model, positions):
        modelListNames = list(model.named_children())
        modelList = list(model.children())
        #buffer = self.buffers()
        # print("\nself.buffers():\n")
        # print(list(model.modules()))
        # print('\n\n')
        # print(list(model.named_buffers()))
        # each pos in pisitions is the position in which the layer sholud be duplicated to make the cnn deeper
        for pos in positions:
            print("\n\nposition:")
            print(pos)

            j = 2 * pos - 2
            conv = modelList[j]
            conv2 = copy.deepcopy(conv)
            modelList.insert(j + 2, conv2)
            convStr = 'conv' + str(pos + 1)
            print('\n\nconvStr:')
            print(convStr)
#            if modelListNames[j+2] == convStr:
                #net is deeper, move all next layers
  #              for k in range(j+2, len(modelList)):
 #           else:
            modelListNames.insert(j + 2, (convStr, conv2))
            bn = modelList[j + 1]
            bn2 = copy.deepcopy(bn)
            bnStr = 'bn' + str(pos + 1)
            modelListNames.insert(j + 3, (bnStr, bn2))
            modelList.insert(j + 3, bn2)
            print("\n> modelListNames:\n")
            print(modelListNames)
        newModel = N2N(10, True)
        # newModel.add_module("buffer",buffer)
        for item in modelListNames:
            j = modelListNames.index(item)
            #print("\nitem: \n")

            if len(item[0]) < 2 and item[0] == 'c':
                #print('\nDrin!!!\n\n')
                itemName = item[0:4]
            elif len(item[0]) < 2 and item[0] == 'b':
                itemName = item[0:2]
            else:
                itemName = item[0]
            #print(itemName)
            newModel.add_module(itemName, modelList[j])
        print(newModel.__dict__.__getitem__('_modules'))
        return newModel
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
