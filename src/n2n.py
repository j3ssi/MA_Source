import copy

import torch
import torch.nn as nn
import math


# Stage0 -> 16
# Stage1 -> 32
# Stage2 -> 64
# Stage3 -> 128
# Stage4 -> 256
class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock, first, model=None):
        super(N2N, self).__init__()
        self.numOfStages = numOfStages
        self.numOfBlocksinStage = numOfBlocksinStage
        self.layersInBlock = layersInBlock
        if first:
            self.module_list = nn.ModuleList()
            # conv1
            conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv1)
            # bn1
            bn1 = nn.BatchNorm2d(16)
            self.module_list.append(bn1)
            firstBlock = True
            for stage in range(0, numOfStages):
                firstLayerInStage = True
                sizeOfLayer = pow(2, stage + 4)
                print("\nStage: ", stage, " ; ", sizeOfLayer)
                for block in range(0, numOfBlocksinStage):
                    if firstLayerInStage and not firstBlock:
                        conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                         stride=2)
                        self.module_list.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        self.module_list.append(bn)
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False, stride=1)
                        self.module_list.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        self.module_list.append(bn)
                        conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=1, padding=0, bias=False,
                                         stride=2)
                        self.module_list.append(conv)
                        bn3 = nn.BatchNorm2d(sizeOfLayer)
                        self.module_list.append(bn)
                        firstLayerInStage = False
                    else:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False, stride=1)
                        self.module_list.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        self.module_list.append(bn)
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False, stride=1)
                        self.module_list.append(conv)
                        bn3 = nn.BatchNorm2d(sizeOfLayer)
                        self.module_list.append(bn)
                        firstBlock = False
                        firstLayerInStage = False
            # 18
            avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.module_list.append(avgpool)
            # 19
            self.sizeOfFC = pow(2, numOfStages + 3)
            fc = nn.Linear(self.sizeOfFC, num_classes)
            self.module_list.append(fc)
            self.relu = nn.ReLU(inplace=True)

            for m in self.module_list:
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            print(self)
        else:
            altList = []
            paramList = []
            for name, param in model.named_parameters():
                # print("\nName: {}", name)
                paramList.append(param)
                i = int(name.split('.')[1])
                if i % 2 == 0:
                    altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')

                elif (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
                    altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
                elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
                    altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")

                elif (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
                    altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
                elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
                    altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
                else:
                    assert True, print("Hier fehlt noch was!!")
            # print("\naltList", altList)
            module_list1 = nn.ModuleList()
            for i in range(len(altList)):
                # print("\n>i: ", i)
                name = altList[i]
                param = paramList[i]
                # print("\nName: ", name)
                if 'conv' in name:
                    dims = list(param.shape)
                    in_chs = dims[1]
                    out_chs = dims[0]
                    # Search for the corresponding Conv Module in Module_list
                    k = int(name.split('.')[1].split('v')[1])
                    module = model.module_list[(k - 1) * 2]
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    bias = module.bias if module.bias is not None else False

                    layer = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
                    # print("\n>new Layer: ", layer, " ; ", param.shape)
                    layer.weight = module.weight
                    module_list1.append(layer)

                elif 'bn' in name and not 'bias' in name:
                    layer = nn.BatchNorm2d(paramList[i].shape[0])
                    # print("\n>new Layer: ", layer)
                    module_list1.append(layer)
                elif 'bn' in name and 'bias' in name:
                    # print("\n>Name: ", name, " ; ", k)
                    k = int(name.split('.')[1].split('n')[1])
                    k1 = 2 * (k - 1) + 1
                    # print("\nk1: ", k1)
                    module = model.module_list[k1]
                    module_list1[-1].bias = module.bias
                    module_list1[-1].weight = module.weight
                # else:
                # print('\nelse: ', name)
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            module_list1.append(avgpool)
            module = model.module_list[-1]
            self.sizeOfFC = paramList[-2].shape[1]
            fc = nn.Linear(paramList[-2].shape[1], num_classes)
            fc.weight = module.weight
            fc.bias = module.bias
            module_list1.append(fc)
            self.module_list = module_list1
            self.relu = nn.ReLU(inplace=True)
            # print("\nnew Model: ", self)

    def forward(self, x):
        first = True
        printNet = False
        # conv1
        x = self.module_list[0](x)
        if printNet:
            print("\nI: 0 ; ", self.module_list[0])

        # bn1
        x = self.module_list[1](x)
        if printNet:
            print("\nI: 1 ; ", self.module_list[1])

        _x = self.relu(x)
        i = 2
        for stage in range(0, self.numOfStages):
            for block in range(0, self.numOfBlocksinStage):
                if first and stage > 0:
                    # conv
                    x = self.module_list[i](_x)
                    if printNet:
                        print("\nI: ", i ," ; ", self.module_list[i])
                    i = i + 1
                    # bn
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    x = self.relu(x)
                    # conv
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    # bn
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    #conv
                    if printNet:
                        print("\n_x: ", _x.shape)
                    _x = self.module_list[i](_x)
                    if printNet:
                        print("\n_x: ", _x.shape)
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    # bn
                    _x = self.module_list[i](_x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    if printNet:
                        print("\n_x: ", _x.shape, " : x: ", x.shape)
                    _x = _x + x
                    _x = self.relu(_x)
                    first = False
                else:
                    # conv2
                    x = self.module_list[i](_x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    # bn2
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    x = self.relu(x)
                    # conv3
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    # bn3
                    x = self.module_list[i](x)
                    if printNet:
                        print("\nI: ", i, " ; ", self.module_list[i])
                    i = i + 1
                    _x = _x + x
                    x = self.relu(_x)
                    first = False
            first = True
        if printNet:
            print("\nI: ", i, " ; ", self.module_list[i])

        if isinstance(self.module_list[i], nn.AdaptiveAvgPool2d):
            try:
                x = self.module_list[i](x)
                if printNet:
                    print("\nI: ", i, " ; ", self.module_list[i])
                x = x.view(-1, self.sizeOfFC)
                i = i + 1

            except RuntimeError:
                print("\n \n Oops!!!: ")
                print("AvgPool")
        if isinstance(self.module_list[i], nn.Linear):
            x = self.module_list[i](x)
            if printNet:
                print("\nfc", i, " ; ", x.shape)
        return x

    def getResidualPath(self):
        stagesI = []
        stagesO = []
        first = True
        i = 0
        for stage in range(0, self.numOfStages):
            stagesI.append([])
            stagesO.append([])
            if first and stage == 0:
                if i % 2 == 0:
                    stagesO[stage].append(n(int(i + 2 / 2)))
                    print("\nI: ", i)
                    i = i + 1
                else:
                    stagesI[stage].append(n(int(i + 2 / 2)))
                    print("\nI: ", i)
                    i = i + 1
            elif first and stage > 0:
                if i % 2 == 1:
                    stagesO[stage].append(n(int(i + 2 / 2)))
                    i = i + 1
                else:
                    stagesI[stage].append(n(int(i + 2 / 2)))
                    i = i + 1

            for block in range(0, self.numOfBlocksinStage):
                for layer in range(0, self.layersInBlock):
                    if i%2 == 0:
                        stagesO[stage].append(n(int(i+2/2)))
                        print("\nI: ", i)
                        i = i + 1
                    else:
                        stagesI[stage].append(n(int(i + 2 / 2)))
                        print("\nI: ", i)
                        i = i + 1

        print("\nStagesI: ", stagesI)
        print("\nStagesO: ", stagesO)
        return stagesI ,stagesO

    def getShareSameNodeLayers(self):
        sameNode = []
        first = True
        i = 1
        for stage in range(0, self.numOfStages):
            for block in range(0, self.numOfBlocksinStage):
                for layer in range(0, self.layersInBlock):
                    if i%2 == 1:
                        sameNode.append((n(i), n(i+1)))
                        i = i+1


        print("\nSame Node: ", sameNode)
        return sameNode


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

    return model


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features




def n(name):
    if isinstance(name, int):
        return 'module.conv' + str(name) + '.weight'
    else:
        return 'module.' + name + '.weight'


def getRmLayers(name, dataset):
    pass
