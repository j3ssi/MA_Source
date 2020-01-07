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

        conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn1 = nn.BatchNorm2d(16)

        #1
        conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn2 = nn.BatchNorm2d(16)
        conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn3 = nn.BatchNorm2d(16)

        # 2
        conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn4 = nn.BatchNorm2d(16)
        conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn5 = nn.BatchNorm2d(16)

        # 3
        conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn6 = nn.BatchNorm2d(16)
        conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn7 = nn.BatchNorm2d(16)

        # 4
        conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn8 = nn.BatchNorm2d(16)
        conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        bn9 = nn.BatchNorm2d(16)

        # 5
        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fc = nn.Linear(16, num_classes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        l = [conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4, conv5, bn5, conv6, bn6, conv7, bn7, conv8, bn8, conv9, bn9, avgpool, fc]
        #print(l)
        self.module_list = nn.ModuleList(l)
        #print("\n\n> moduleList:\n")
        #print(self.module_list)

    def forward(self, x):

        x = self.module_list[0](x)
        x = self.module_list[1](x)
        _x = self.relu(x)
        i = 2
        while i > 0:
            if isinstance(self.module_list[i],nn.AdaptiveAvgPool2d((1,1))):
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
            #         except RuntimeError:
            #             print("\n \n Oops!!! \n \n \n")
            #             print(convStr)
            # if convStr not in names:
            #
            #     x = self.avgpool(_x)
            #
            #     x = self.fc(x)
            #     return x
            # # find the module with name convStr
            # for name, module in self.named_modules():
            #     if name == convStr:
            #
            # bnStr = 'bn' + str(i)
            # for name, module in self.named_modules():
            #     if name == bnStr:
            #         try:
            #             x = module.forward(x)
            #             break
            #         except RuntimeError:
            #             print("\n \n Oops!!! \n \n \n")
            #             print(bnStr)
            # x = self.relu(x)
            # i = i + 1
            # convStr = 'conv' + str(i)
            # for name, module in self.named_modules():
            #     if name == convStr:
            #         try:
            #             x = module.forward(x)
            #             break
            #         except RuntimeError:
            #             print("\n \n Oops!!! \n \n \n")
            #             print(convStr)
            #
            # bnStr = 'bn' + str(i)
            #
            # for name, module in self.named_modules():
            #     if name == bnStr:
            #         try:
            #             x = module.forward(x)
            #             break
            #         except RuntimeError:
            #             print("\n \n Oops!!! \n \n \n")
            #             print(bnStr)
            # try:
            #     _x = _x + x
            # except RuntimeError:
            #     print("\n \n Oops!!  \n \n \n")
            #     print('_x = _x + x')
            #
            # _x = self.relu(_x)
            # i = i + 1

    def deeper(self, model, positions):
        modelList = list(model.children())
        print("\nself.modules():\n")
        #print(list(model.modules()))
        print(list(self.modules()))
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
            #print('\n\nconvStr:')
            #print(convStr)
           # if modelListNames[j+2] == convStr:
                #net is deeper, move all next layers first and then insert new conv layer
                #insert is theroatically fine for this but the names of the layer are not updated
            #    for k in range(j+2, len(modelList)):
             #       name, module = modelListNames.__getitem__(k)
              #      if 'conv' in name:
               #         newName = 'conv' + str(k/2+1)
                #    elif 'bn' in name:
                 #       newName = 'bn' + str((k-1)/2+1)
                  #  modelListNames[k] = (newName,module)
            #modelListNames.insert(j + 2, (convStr, conv2))
            bn = modelList[j + 1]
            bn2 = copy.deepcopy(bn)
            bnStr = 'bn' + str(pos + 1)
            #modelListNames.insert(j + 3, (bnStr, bn2))
            modelList.insert(j + 3, bn2)
            #print("\n> modelListNames:\n")
            #print(modelListNames)
        newModel = nn.ModuleList(*modelList)
        # newModel.add_module("buffer",buffer)
        #for item in modelListNames:
        #    j = modelListNames.index(item)
        #    #print("\nitem: \n")

#            if len(item[0]) < 2 and item[0] == 'c':
 #               #print('\nDrin!!!\n\n')
  #              itemName = item[0:4]
   #         elif len(item[0]) < 2 and item[0] == 'b':
    #            itemName = item[0:2]


        print("\nnew Model:\n")
        print(list(newModel.modules()))
        print()
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
