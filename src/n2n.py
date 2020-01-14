import copy

import torch
import torch.nn as nn
import math


class N2N(nn.Module):

    def __init__(self, num_classes, num_residual_blocks, first, model=None):
        super(N2N, self).__init__()
        if first:
            self.module_list = nn.ModuleList()
            conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv1)
            bn1 = nn.BatchNorm2d(16)
            self.module_list.append(bn1)

            for block in range(num_residual_blocks):
                conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
                self.module_list.append(conv2)
                bn2 = nn.BatchNorm2d(16)
                self.module_list.append(bn2)
                conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
                self.module_list.append(conv3)
                bn3 = nn.BatchNorm2d(16)
                self.module_list.append(bn3)

            # 18
            conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv1)
            bn1 = nn.BatchNorm2d(16)
            self.module_list.append(bn1)

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

        # else:
        #     altList = []
        #     paramList = []
        #     for name, param in model.named_parameters():
        #         #print("\nName: {}", name)
        #         paramList.append(param)
        #         i = int(name.split('.')[1])
        #         if i % 2 == 0:
        #             altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')
        #
        #         if (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
        #             altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
        #         elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
        #             altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")
        #
        #         if (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
        #             altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
        #         elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
        #             altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
        #
        #     #print("\naltList", altList)
        #     module_list1 = nn.ModuleList()
        #     for i in range(len(altList)):
        #         name = altList[i]
        #         param = paramList[i]
        #         #print("\nName: ", name)
        #         if 'conv' in name:
        #             dims = list(param.shape)
        #             in_chs = dims[1]
        #             if 'conv' in altList[i+2] :
        #                 out_chs = paramList[i+2].shape[1]
        #             # Search for the corresponding Conv Module in Module_list
        #             k = int(name.split('.')[1].split('v')[1])
        #             module = model.module_list[(k - 1) * 2]
        #             kernel_size = module.kernel_size
        #             stride = module.stride
        #             padding = module.padding
        #             bias = module.bias if module.bias != None else False
        #
        #             layer = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding,
        #                               bias=bias)
        #             print("\n>new Layer: ", layer, " ; ", param.shape)
        #             layer.weight = module.weight
        #             module_list1.append(layer)
        #
        #         elif 'bn' in name and not 'bias' in name:
        #             if'conv' in (altList[i+2]):
        #                 layer = nn.BatchNorm2d(paramList[i+2].shape[1])
        #                 print("\n>new Layer: ", layer)
        #             else:
        #                 layer = nn.BatchNorm2d(paramList[i].shape[0])
        #             module_list1.append(layer)
        #         elif 'bn' in name and 'bias' in name:
        #             module_list1[-1].bias
        #         else:
        #             print('\nelse: ', name)
        #
        #     avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #     module_list1.append(avgpool)
        #     fc = nn.Linear(16, num_classes)
        #     module_list1.append(fc)
        #     self.module_list = module_list1
        #     self.relu = nn.ReLU(inplace=True)
        #     print("\nnew Model: ", self)

    def forward(self, x):
        odd = False
        first = True
        bn = False
        _x = None
        printNet = False
        i = 0
        for module in self.module_list:
            if isinstance(module, nn.AdaptiveAvgPool2d):
                try:
                    x = module(_x)
                    x = x.view(-1, 16)

                    if printNet:
                        print("\navgpool", i)
                        i = i + 1
                except RuntimeError:
                    print("\n \n Oops!!!: ")
                    print("AvgPool")
            elif isinstance(module, nn.Linear):
                x = module(x)
                if printNet:
                    print("\nfc", i)
                return x
            else:
                if first and not bn:
                    x = module(x)
                    bn = True
                    if printNet:
                        print("\nFirst conv", i)
                        i = i + 1
                elif first and bn:
                    x = module(x)
                    _x = self.relu(x)
                    if printNet:
                        print("\nFirst bn", i)
                        i = i + 1
                    first = False
                    bn = False
                else:
                    if not odd and not bn:
                        x = module(_x)
                        if printNet:
                            print('\nconv', i)
                            i = i + 1
                        bn = True
                    elif not odd and bn:
                        x = module(x)
                        x = self.relu(x)
                        if printNet:
                            print("\nbn", i)
                            i = i + 1
                        odd = True
                        bn = False
                    else:
                        if not bn:
                            x = module(x)
                            bn = True
                            if printNet:
                                print('Odd conv', i)
                                i = i + 1
                        elif bn:
                            x = module(x)
                            _x = _x + x
                            _x = self.relu(_x)
                            odd = False
                            bn = False
                            if printNet:
                                print('Odd bn', i)
                                i = i + 1


def deeper(model, optimizer, positions):
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
    listI.append(n('fc'))
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
    print("\nSame Node: ", sameNode)
    return sameNode


def n(name):
    if isinstance(name, int):
        return 'module.conv' + str(name) + '.weight'
    else:
        return 'module.' + name + '.weight'


def getRmLayers(name, dataset):
    pass
