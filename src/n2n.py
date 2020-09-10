import copy

import numpy
import torch
import torch as th
import torch.nn as nn
import math
import numpy as np


class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock,
                 first, bottleneck, widthofFirstLayer=16, model=None, archNums=None, widthOfLayers=None):
        super(N2N, self).__init__()
        # print(f'width: {widthOfLayers}')
        # self.device = torch.device("cuda:0")
        self.numOfStages = numOfStages
        self.oddLayers = []
        self.numOfBlocksinStage = numOfBlocksinStage
        self.bottleneck = bottleneck
        self.layersInBlock = layersInBlock
        if widthOfLayers is not None:
            self.widthofFirstLayer = widthOfLayers[0]
            self.widthofLayers = widthOfLayers
            # print(f'width: {self.widthofFirstLayer}')
        else:
            self.widthofFirstLayer = widthofFirstLayer
            self.widthofLayers = []
            s = widthofFirstLayer

            print(f'numoFStages: {numOfStages}')

            for stage in range(0, numOfStages):
                self.widthofLayers.append(s)
                s *= 2

        if first:
            self.archNums = [[]]
            for s in range(0, self.numOfStages):
                # print("\nS: ", s, " ; ", self.numOfStages)
                for b in range(0, self.numOfBlocksinStage[s]):
                    # print(f'b: {b}')
                    if b == 0 and s > 0:
                        self.archNums[s].append(self.layersInBlock + 1)
                    else:
                        self.archNums[s].append(self.layersInBlock)

                if s != (self.numOfStages - 1):
                    self.archNums.append([])
            # print("\nArch Num: ", self.archNums)

            self.module_list = nn.ModuleList()

            # first Layer
            # conv1
            conv0 = nn.Conv2d(3, self.widthofFirstLayer, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv0)
            # bn1
            bn1 = nn.BatchNorm2d(self.widthofFirstLayer)
            self.module_list.append(bn1)
            if self.bottleneck:
                for stage in range(0, numOfStages):
                    first = True
                    sizeOfLayer = pow(2, stage + 4)
                    # print("\nStage: ", stage, " ; ", sizeOfLayer)
                    # CONV 1
                    if stage == 0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)
                    else:
                        conv = nn.Conv2d(sizeOfLayer * 2, sizeOfLayer, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)
                    # print(f'list length: {len(self.module_list)}')
                    # print(f'conv: {conv}')
                    self.module_list.append(conv)
                    # bn1
                    bn = nn.BatchNorm2d(sizeOfLayer)
                    self.module_list.append(bn)
                    # conv2
                    conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1,
                                     bias=False,
                                     stride=1)
                    self.module_list.append(conv)
                    # bn 2
                    bn = nn.BatchNorm2d(sizeOfLayer)
                    self.module_list.append(bn)

                    conv = nn.Conv2d(sizeOfLayer, sizeOfLayer * 4, kernel_size=1, padding=0,
                                     bias=False,
                                     stride=1)
                    self.module_list.append(conv)
                    bn = nn.BatchNorm2d(sizeOfLayer * 4)
                    self.module_list.append(bn)
                    if stage == 0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer * 4, kernel_size=3, padding=1,
                                         bias=False,
                                         stride=1)
                    else:
                        conv = nn.Conv2d(sizeOfLayer * 2, sizeOfLayer * 4, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)

                    self.module_list.append(conv)
                    bn = nn.BatchNorm2d(sizeOfLayer * 4)
                    self.module_list.append(bn)

                    # print(f'archNums: {len(self.archNums[stage - 1])}')
                    for i in range(0, len(self.archNums[stage - 1]) - 1):
                        j = 0
                        while j < self.archNums[stage - 1][i + 1]:
                            # print(f'self.archNums[stage-1][i+1]:{self.archNums[stage - 1][i + 1]}')
                            if (j == 0):
                                conv = nn.Conv2d(sizeOfLayer * 4, sizeOfLayer, kernel_size=1, padding=0,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                j = j + 1
                            elif (j + 1) % self.archNums[stage - 1][i + 1] != 0:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                j = j + 1
                            else:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer * 4, kernel_size=1, padding=0,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(4 * sizeOfLayer)
                                self.module_list.append(bn)
                                j = j + 1

                # 18
                self.sizeOfFC = pow(2, self.numOfStages + 5)

                # conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                #                  stride=1)
                # self.module_list.append(conv)
                # bn = nn.BatchNorm2d(sizeOfLayer)
                # self.module_list.append(bn)

                # print("\n self sizeofFC: ",self.sizeOfFC)
                avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.module_list.append(avgpool)
                # 19
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
            else:
                # print(f'ohne Bottleneck!')
                firstLayer = True
                for stage in range(0, numOfStages):
                    firstBlockInStage = True
                    sizeOfLayer = pow(2, stage)
                    if widthOfLayers is not None:
                        sizeOfLayer = widthOfLayers[stage]
                    else:
                        sizeOfLayer *= self.widthofFirstLayer
                    # print(f'stage: {stage}; sizeof Layers: {sizeOfLayer}')
                    # print("\nStage: ", stage, " ; ", sizeOfLayer)
                    for block in range(0, len(self.archNums[stage])):
                        i = 0
                        while i < self.archNums[stage][block]:
                            # print(f'i : {i}')
                            if firstBlockInStage and not firstLayer and i == 0:
                                if widthOfLayers is not None:
                                    conv = nn.Conv2d(widthOfLayers[stage - 1], sizeOfLayer, kernel_size=3, padding=1,
                                                     bias=False,
                                                     stride=2)

                                else:
                                    conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                                     bias=False,
                                                     stride=2)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                i = i + 1

                            elif firstBlockInStage and not firstLayer and (i + 1) % self.archNums[stage][block] == 0:

                                if widthOfLayers is not None:
                                    conv = nn.Conv2d(widthOfLayers[stage - 1], sizeOfLayer, kernel_size=3, padding=1,
                                                     bias=False,
                                                     stride=2)

                                else:
                                    conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                                     bias=False,
                                                     stride=2)
                                self.module_list.append(conv)

                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                i = i + 1

                                firstBlockInStage = False

                            elif firstBlockInStage and not firstLayer:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                i = i + 1

                            else:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                i = i + 1

                    firstLayer = False

                    # 18

                # conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                #                  stride=1)
                # self.module_list.append(conv)
                # bn = nn.BatchNorm2d(sizeOfLayer)
                # self.module_list.append(bn)

                # print("\n self sizeofFC: ",self.sizeOfFC)
                avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.module_list.append(avgpool)
                # 19
                fc = nn.Linear(sizeOfLayer, num_classes)
                self.module_list.append(fc)
                self.relu = nn.ReLU(inplace=True)

                for m in self.module_list:
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()
            print(f'Modell Erstellung')
            print(self)
            self.sameNode, self.oddLayers = buildShareSameNodeLayers(self.module_list, self.numOfStages, self.archNums)
            self.stageI, self.stageO = buildResidualPath(self.module_list, self.numOfStages, self.archNums)
            # print(f'sameNode: {self.sameNode}')
        else:
            self.archNums = archNums

            self.sameNode = model.sameNode
            self.stageI = model.stageI
            self.stageO = model.stageO
            # print(f'Archnums: {self.archNums}')
            module_list = model.module_list
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
            # print(altList)
            del model
            self.module_list = nn.ModuleList()

            # print("\naltList", altList)
            for i in range(0, len(altList)):
                # print("\n>i: ", i)
                name = altList[i]
                param = paramList[i]
                # print("\nName: ", name)
                if 'conv' in name:
                    dims = list(param.shape)
                    # print("Name, Dims: ", name, " ; ", dims)
                    in_chs = dims[1]
                    out_chs = dims[0]
                    # Search for the corresponding Conv Module in Module_list
                    k = int(name.split('.')[1].split('v')[1])
                    # print("\nK: ", k, " ; ", (k - 1) * 2, "; in,out: ", in_chs, " ; ", out_chs)
                    module = module_list[(k - 1) * 2]
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    bias = module.bias if module.bias is not None else False

                    layer = nn.Conv2d(in_chs, out_chs, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=bias)
                    if printName:
                        print("\n>new Layer: ", layer, " ; ", param.shape)
                        print("\nWeight Shape: ", module.weight.shape)
                    layer.weight.data = module.weight.data
                    self.module_list.append(layer)

                elif 'bn' in name and 'bias' not in name:
                    layer = nn.BatchNorm2d(paramList[i].shape[0])
                    if printName:
                        print("\n>new Layer: ", layer)
                    self.module_list.append(layer)
                elif 'bn' in name and 'bias' in name:
                    if printName:
                        print("\n>Name: ", name, " ; ", k)
                    k = int(name.split('.')[1].split('n')[1])
                    k1 = 2 * (k - 1) + 1
                    # print("\nk1: ", k1)
                    module = module_list[k1]
                    self.module_list[-1].bias.data = module.bias.data
                    self.module_list[-1].weight.data = module.weight.data
                # else:
                # print('\nelse: ', name)
            module = module_list[-1]
            # print(f'module: {module}')
            self.sizeOfFC = module.weight.shape[1]
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.module_list.append(avgpool)
            # <if printName:
            # print("\n self sizeofFC: ", self.sizeOfFC)
            fc = nn.Linear(module.weight.shape[1], num_classes)
            if printName:
                print("\nLinear: ", fc)
            fc.weight.data = module.weight.data
            fc.bias.data = module.bias.data
            self.module_list.append(fc)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print(f'ArchNums: {self.archNums}')
        # First layer
        printNet = False
        if printNet:
            print("\nX Shape: ", x.shape)
        # conv1
        x = self.module_list[0](x)
        if printNet:
            print("\nI: 0 ; ", self.module_list[0])
            print("\nX Shape: ", x.shape)
        # bn1
        x = self.module_list[1](x)
        if printNet:
            print("\nI: 1 ; ", self.module_list[1])
            print("\nX Shape: ", x.shape)

        _x = self.relu(x)
        j = 2
        notfirstLayer = False
        # try:
        for stage in range(0, self.numOfStages):
            if printNet:
                print("\n\nStage: ", stage)
            archNum = self.archNums[stage]
            firstBlockInStage = True
            for block in range(0, len(archNum)):
                if self.bottleneck:
                    if firstLayerInStage:
                        if printNet:
                            print("\n\n\tBlock: ", block)
                        i = 0
                        layerInThisBlock = archNum[block]
                        # conv2(2)
                        x = self.module_list[j](_x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn3(2)
                        x = self.module_list[j](x)
                        if printNet:
                            print(f'J:  {j}; {self.module_list[j]}')
                        j = j + 1
                        x = self.relu(x)

                        # conv4 (3)
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn5 (3)
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1
                        x = self.relu(x)

                        # conv6 (4)
                        x = self.module_list[j](x)
                        if printNet:
                            print(f'J:  {j}; {self.module_list[j]}')
                        j = j + 1

                        # bn7 (4)
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # conv8 (5)
                        _x = self.module_list[j](_x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn9 (5)
                        _x = self.module_list[j](_x)
                        if printNet:
                            print(f'J:  {j}; {self.module_list[j]}')
                        j = j + 1
                        i = i + 1
                        _x = x + _x
                        _x = self.relu(_x)
                        firstLayerInStage = False
                    else:
                        if printNet:
                            print("\n\n\tBlock: ", block)

                        i = 0
                        layerInThisBlock = archNum[block]
                        while i < layerInThisBlock:
                            # conv
                            if i == 0:
                                x = self.module_list[j](_x)
                            else:
                                x = self.module_list[j](x)
                            if printNet:
                                print("\n conv J: ", j, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            x = self.module_list[j](x)
                            if printNet:
                                print("\nbn: ", j, " ; ", self.module_list[j])
                            j = j + 1

                            x = self.relu(x)

                            if ((i + 1) % layerInThisBlock) == 0:
                                _x = _x + x
                                _x = self.relu(_x)

                else:
                    if printNet:
                        print(f'Block: {block}; j: {j}')
                    i = 0
                    layerInThisBlock = archNum[block]
                    while i < layerInThisBlock:
                        if i == 0:
                            # conv
                            x = self.module_list[j](_x)
                            if printNet:
                                print("\nLayer of new Stage i: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            x = self.module_list[j](x)
                            if printNet:
                                print("\ni: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1
                            i = i + 1

                            # relu
                            x = self.relu(x)
                        elif ((i + 1) % layerInThisBlock == 0) and firstBlockInStage and notfirstLayer:
                            # conv
                            _x = self.module_list[j](_x)
                            if printNet:
                                print("\nj: ", j, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            _x = self.module_list[j](_x)
                            if printNet:
                                print("\nj: ", j, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)

                            j = j + 1
                            i = i + 1
                            firstBlockInStage = False
                            # print(f'j vor Add: {j}')
                            _x = x + _x
                            _x = self.relu(_x)

                        elif ((i + 1) % layerInThisBlock == 0):
                            # conv
                            x = self.module_list[j](x)
                            j = j + 1

                            if printNet:
                                print("\n i: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)

                            # bn
                            x = self.module_list[j](x)
                            if printNet:
                                print("\nShortcutLayer i: ", i, " ; ", self.module_list[j])
                            j = j + 1
                            i = i + 1

                            _x = _x + x
                            _x = self.relu(_x)

                        else:
                            # conv
                            x = self.module_list[j](x)
                            if printNet:
                                print("\ni: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            x = self.module_list[j](x)
                            if printNet:
                                print("\ni: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1
                            x = self.relu(x)
                            i = i + 1
                    firstBlockInStage = False
                    notfirstLayer = True
        # except RuntimeError:
        #     print(f'Except')
        #     print("\nJ: ", j, " ; ", self.module_list[j])
        #     print("\nX Shape: ", x.shape)

        if printNet:
            print("\nX Shape: ", x.shape)

        if isinstance(self.module_list[j], nn.AdaptiveAvgPool2d):
            try:
                x = self.module_list[j](_x)
                if printNet:
                    print("\nJ: ", j, " ; ", self.module_list[j])
                    print("\n\n X Shape 1: ", x.shape)
                x = x.view(x.shape[0], x.shape[1])
                if printNet:
                    print("\n\n X Shape 2: ", x.shape)
            except RuntimeError:
                print("\n \n Oops!!!: ")
                print("AvgPool")
        elif isinstance(self.module_list[j], nn.Conv2d):
            print(f'Sollte nicht Conv sondern AvgPool sein {j}; {self.archNums}')
        elif isinstance(self.module_list[j], nn.Linear):
            print(f'Sollte nicht Linear sondern AvgPool sein {j}')
        elif isinstance(self.module_list[j], nn.BatchNorm2d):
            print(f'Sollte nicht Bn sondern AvgPool sein {j}')
        else:
            print("\n \nElse Oops!!!: ")
            print("AvgPool")

        j = j + 1
        if isinstance(self.module_list[j], nn.Linear):
            x = self.module_list[j](x)
            if printNet:
                print("\nJ: ", j, " ; ", self.module_list[j])
                print("\nfc: ", x.shape)
        else:
            print("\n \n Oops!!!: ")
            print("Linear")
        return x

    def delete(self, model, index):
        printNet = False
        # print(f'Index: {index}')
        index1 = int(index / 2 + 1)
        # print(f'Index1: {index1}')
        j = 2
        blockBegin = []
        for stage in range(0, self.numOfStages):
            if printNet:
                print("\n\nStage: ", stage)
            archNum = self.archNums[stage]
            firstBlockInStage = True
            for block in range(0, len(archNum)):
                blockBegin.append(j)
                if printNet:
                    print("\n\n\tBlock: ", block)
                i = 0
                k = j
                layerInThisBlock = archNum[block]
                while i < layerInThisBlock:
                    # print(f'j: {j}; k: {k}')
                    if j == index1:
                        numDelete = self.archNums[stage][block]
                        stageDelete = stage
                        blockDelete = block
                        # print(f'numDelete: {numDelete}')
                    j = j + 1
                    i = i + 1
        # print(f'blockBeginn: {blockBegin}')

        module_list = nn.ModuleList()
        deleteModule = True
        thisBlockBeginn = 0
        for layers in range(0, (len(self.module_list) - 2 * numDelete)):
            if layers < index:
                module_list.append(self.module_list[layers])
                # print(f'Kopiere {layers}: {module_list[layers]}')
            elif layers >= index:
                module_list.append(self.module_list[layers + 2 * numDelete])
                # print(
                #    f'Ersetze {layers} gegen {layers + 2 * numDelete}: {self.module_list[layers]} gegen {self.module_list[layers + 2 * numDelete]}')
        # print(f'archnums vorher: {self.archNums}')
        self.archNums[stageDelete][blockDelete] = 0
        self.archNums[stageDelete].remove(0)
        self.module_list = module_list
        self.sameNode, self.oddLayers = buildShareSameNodeLayers(module_list, self.numOfStages, self.archNums)
        # print(f'sameNode: {self.sameNode}')
        self.stageI, self.stageO = buildResidualPath(self.module_list, self.numOfStages, self.archNums)
        tempStage = []

        # print(f'stageI: {self.stageI}')
        # print(f'stageO: {self.stageO}')

        # print(f'archnums nachher: {self.archNums}')
        # print(self)
        return model

    def getResidualPath(self):
        return self.stageI, self.stageO

    def getShareSameNodeLayers(self):
        return self.sameNode

    """
    Convert all layers in layer to its wider version by adapting next weight layer and possible batch norm layer in btw.
    layers = 'conv 3, conv6'
    """

    def wider(self, stage, delta_width, out_size=None, weight_norm=True, random_init=True, addNoise=True):
        print(f'Stage: {stage}')
        # get names for modules
        altList = []
        paramList = []
        printName = False
        for name, param in self.named_parameters():
            paramList.append(param)
            i = int(name.split('.')[1])

            if i % 2 == 0:
                altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')
                if printName:
                    print("\nI:", i, " ; ", altList[-1])
            elif (i % 2 == 1) and ('weight' in name) and (i < (len(self.module_list) - 2)):
                altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
                if printName:
                    print("\nI:", i, " ; ", altList[-1])
            elif (i % 2 == 1) and ('weight' in name) and (i > (len(self.module_list) - 3)):
                altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")
                if printName:
                    print("\nI:", i, " ; ", altList[-1])
            elif (i % 2 == 1) and ('bias' in name) and (i < (len(self.module_list) - 1)):
                altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
                if printName:
                    print("\nI:", i, " ; ", altList[-1])
            elif (i % 2 == 1) and ('bias' in name) and (i > (len(self.module_list) - 2)):
                altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
                if printName:
                    print("\nI:", i, " ; ", altList[-1])
            else:
                assert True, print("Hier fehlt noch was!!")

        # fill lists with numbers of input or output numbers for each stage
        residualListI = []
        residualListO = []

        for index in range(0, len(altList)):
            if 'conv' in altList[index]:
                width = paramList[index].size()[1]
                # print(f'width: {width}')
                if self.widthofLayers.count(width) > 0:
                    tobestage = self.widthofLayers.index(width) + 1
                    # print(f'stage: {stage}')
                    if tobestage == stage:
                        num = int(altList[index].split('.')[1].split('v')[1])
                        residualListI.append(num)

                width = paramList[index].size()[0]
                if self.widthofLayers.count(width) > 0:
                    tobestage = self.widthofLayers.index(width) + 1
                    # print(f'stage: {stage}')
                    if tobestage == stage:
                        num = int(altList[index].split('.')[1].split('v')[1])
                        residualListO.append(num)

        tmpListI = copy.copy(residualListI)
        tmpListO = copy.copy(residualListO)
        residualList = sorted(list(set(tmpListI) | set(tmpListO)))

        # fill numpy array with random elemente from original weight
        index = 0
        while index == 0:
            # get next elemente to widen
            j = residualList.pop(0)
            # transform to numbetr in moduleList
            print(f'j: {j}')
            i = 2 * j - 2

            # get modules
            m1 = self.module_list[i]
            w1 = m1.weight.data.clone().cpu().numpy()
            bn = self.module_list[i + 1]
            bnw1 = bn.weight.data.clone().cpu().numpy()
            bnb1 = bn.bias.data.clone().cpu().numpy()
            assert delta_width > 0, "New size should be larger"

            if j in residualListI:
                print(f'Resiudual I')
                old_width = m1.weight.size(1)
                new_width = old_width * delta_width
                print(f'new width: {new_width}; old width: {old_width}')
                dw1 = []
                tracking = dict()
                listindices = []
                for o in range(0, (new_width - old_width)):
                    idx = np.random.randint(0, old_width)
                    # print(f'idx: {idx}')
                    m1list = w1[:, idx, :, :]
                    listindices.append(idx)
                    try:
                        tracking[idx].append(o)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o)
                    # TEST:random init for new units
                    if random_init:
                        n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n),
                                                  size=(w1.shape[0], new_width - old_width, w1.shape[2], w1.shape[3]))
                        print(f'dw1: {dw1.shape}')
                    else:
                        dw1.append(m1list)

                # print(f'dw1:{dw1}')
                if not random_init:
                    dw1x = np.array(dw1)
                    dw1x = np.transpose(dw1x, [1, 0, 2, 3])
                    w1 = np.concatenate((w1, dw1x), axis=1)

                else:
                    w1 = np.concatenate((w1, dw1), axis=1)

                # print(f'shape after concat: {w1.shape}')

                m1.in_channels = new_width

                std = w1.std()
                if addNoise:
                    noise = np.random.normal(scale=5e-2 * std,
                                             size=w1.shape)
                    w1 += noise

            if j in residualListO:
                print(f'Residual O')
                old_width = m1.weight.size(0)
                new_width = old_width * delta_width
                print(f'old width1: {old_width}; new width: {new_width}')

                dw1 = []
                dbn1w = []
                dbn1rv = []
                dbn1rm = []
                dbn1b = []
                tracking = dict()
                listOfNumbers = []
                listOfRunningMean = []
                listOfRunningVar = []
                for name, buf in self.named_buffers():
                    # print("\nBuffer Name: ", name)
                    if 'running_mean' in name:
                        k = int(name.split('.')[1])
                        if k == (i + 1):
                            mean = buf.clone()
                            listOfRunningMean = mean.cpu().numpy()

                    if 'running_var' in name:
                        k = int(name.split('.')[1])
                        if (k == (i + 1)):
                            var = buf.clone()
                            listOfRunningVar = var.cpu().numpy()

                listindices = []
                # print(f'oldwidth: {old_width} ')
                for o in range(0, (new_width - old_width)):
                    idx = np.random.randint(0, old_width)
                    m1list = w1[idx, :, :, :]
                    listindices.append(idx)
                    try:
                        tracking[idx].append(o)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o)

                    # TEST:random init for new units
                    if random_init:
                        n1 = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n1),
                                                  size=(new_width - old_width, w1.shape[1], w1.shape[2], w1.shape[3]))
                        print(f'dw1: {dw1.shape}')
                    else:
                        dw1.append(m1list)

                    dbn1 = listOfRunningMean[idx]
                    # print(f'length of dbn1: {dbn1}')
                    dbn1rm.append(dbn1)
                    dbn1 = listOfRunningVar[idx]
                    dbn1rv.append(dbn1)
                    dbn1w.append(bnw1[idx])
                    dbn1b.append(bnb1[idx])
                    bn.num_features = new_width
                # print(f'indices: {listindices}')
                # print(f'tracking dict: {tracking}')
                ct = {}
                for key, dif_k in tracking.items():
                    # print(f'key: {key}; difk: {dif_k}')
                    dictcounter = len(dif_k)
                    ct.update({key: dictcounter})
                # print(f'ct: {ct}')
                if not random_init:
                    for idx in range(len(listindices)):
                        c = dw1[idx]

                        # print(f'c:{c}')
                        for k in range(len(c)):
                            e = c[k]
                            # print(f'c[k]: {c[k]}')
                            for l in range(len(e)):
                                # print(f' before e[l]: {e[l]}')
                                f = e[l]
                                for m in range(len(f)):
                                    f[m] = f[m] / ct.get(listindices[idx])
                                # print(f' after e[l]: {e[l]}')

                dw1x = np.array(dw1)

                w1 = np.concatenate((w1, dw1x), axis=0)

                rm = torch.FloatTensor(dbn1rm).cuda()
                rm1 = torch.FloatTensor(listOfRunningMean).cuda()
                nbn1rm = torch.cat((rm1, rm), dim=0)

                rv = torch.FloatTensor(dbn1rv).cuda()
                rv1 = torch.FloatTensor(listOfRunningVar).cuda()
                nbn1rv = torch.cat((rv1, rv))

                dbn1wa = torch.FloatTensor(dbn1w).cuda()
                nbn1w = torch.cat((bn.weight, dbn1wa))

                dbn1x = torch.FloatTensor(dbn1b).cuda()
                nbn1b = torch.cat((bn.bias.data, dbn1x))

                m1.out_channels = new_width
                x = w1.std()
                if addNoise:
                    noise = np.random.normal(scale=5e-2 * x,
                                             size=(w1.shape))
                    w1 += noise

                if bn is not None:
                    bn.running_var = nbn1rv
                    bn.running_mean = nbn1rv
                    if bn.affine:
                        bn.weight.data = nbn1w
                        bn.bias.data = nbn1b

            m1x = torch.FloatTensor(w1).cuda()
            m1x.requires_grad = True
            m1.weight = torch.nn.Parameter(m1x)

            if len(residualList) == 0:
                index = 1

        # print(f'Bis Hier!')
        # print(f'stage: {stage}')
        # print(f'self num of stages: {self.numOfStages}')
        if int(stage) == int(self.numOfStages) and not random_init:
            module = self.module_list[-1]
            w1 = module.weight.data.clone().cpu().numpy()

            print(f'size: {w1.size}')

            old_width = w1.shape[1]
            new_width = old_width * delta_width
            print(f'old width: {old_width}')
            dw1 = []
            tracking = dict()
            listOfNumbers = []
            listindices = []
            for o in range(0, (new_width - old_width)):
                idx = np.random.randint(0, old_width)
                print(f'idx: {idx}')
                m1list = w1[:, idx]
                listindices.append(idx)

                try:
                    tracking[idx].append(o)
                except:
                    tracking[idx] = []
                    tracking[idx].append(o)

                # TEST:random init for new units
                if random_init:
                    n = module.in_features * module.out_features
                    dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n),
                                              size=(new_width - old_width, module.out_features))
                    # if m2.weight.dim() == 4:
                    #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                    # elif m2.weight.dim() == 5:
                    #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                    # elif m2.weight.dim() == 2:
                    #    n2 = m2.out_features * m2.in_features
                    # dw1.select(0, i).normal_(0, )
                    # dw2.select(0, i).normal_(0, np.sqrt(2. / n2))
                else:
                    dw1.append(m1list)
                    # dw2.append(m2list)
                    # dw1.select(0, i).copy_(w1.select(0, idx).clone())
                    # dw2.select(0, i).copy_(w2.select(0, idx).clone())

            dw1x = np.transpose(dw1, [1, 0])
            dw1y = np.concatenate((w1, dw1x), axis=1)
            w1 = torch.FloatTensor(dw1y).cuda()
            w1.requires_grad = True

            module.in_features = new_width
            module.weight = torch.nn.Parameter(w1)

            # print(f'Model after wider: {self}')
        elif int(stage) == int(self.numOfStages) and random_init:
            module = self.module_list[-1]
            w1 = module.weight.data.clone().cpu().numpy()

            print(f'size: {w1.size}')

            old_width = w1.shape[1]
            new_width = old_width * delta_width
            print(f'old width: {old_width}')
            dw1 = []
            tracking = dict()
            listOfNumbers = []
            listindices = []
            n = module.in_features * module.out_features
            dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n), size=(module.out_features, new_width - old_width))

            print(f'Size w1: {w1.shape}; dw1 size: {dw1.shape}')
            dw1y = np.concatenate((w1, dw1), axis=1)
            w1 = torch.FloatTensor(dw1y).cuda()
            w1.requires_grad = True

            module.in_features = new_width
            module.weight = torch.nn.Parameter(w1)

            # print(f'Model after wider: {self}')

        # print(self)
        return self

    def deeper1(self):
        # make each block with plus two layers (conv +batch) deeper
        printDeeper = True
        j = 2
        notfirstStage = False
        for stage in range(0, self.numOfStages):
            if printDeeper:
                print("\n\nStage: ", stage)
            archNum = self.archNums[stage]
            firstBlockInStage = True

            for block in range(0, len(archNum)):
                if printDeeper:
                    print("\n\n\tBlock: ", block)
                firstBlockInStage = False
                module = self.module_list[j]
                i0 = module.weight.size(0)
                i1 = module.weight.size(1)
                i2 = module.weight.size(2)
                i3 = module.weight.size(3)
                if printDeeper:
                    print(f'size:{i0}, {i1}, {i2}, {i3}; j: {j}')
                dw1 = numpy.ones((i0, i0, i2, i3), dtype=numpy.float32)
                w1 = torch.FloatTensor(dw1)
                w1.requires_grad = True
                kernel_size = i2
                stride = 1
                padding = 1
                bias = module.bias if module.bias is not None else False

                layer = nn.Conv2d(i0, i0, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=bias)

                layer.weight = torch.nn.Parameter(w1)
                j = j + 2
                self.module_list.insert(j, layer)

                layer2 = nn.BatchNorm2d(i0)
                archNum[block] += 1
                layerInThisBlock = archNum[block]
                j = j + 1
                self.module_list.insert(j, layer2)
                i = 3
                j = j + 1
                if printDeeper:
                    print(f'j: {j}; i: {i}')
                while i < layerInThisBlock:
                    i = i + 1
                    j = j + 2
                j= j +2
            notfirstStage = True

        # noise = torch.Tensor(conv2.weight.shape).random_(0, 1).to(self.device)
        # noise = torch.rand(0,0.5)
        print(f'archNums: {self.archNums}')
        return self


def compare(layer, oddLayer):
    i1 = int(layer.split('.')[1].split('v')[1])
    i2 = int(oddLayer.split('.')[1].split('v')[1])
    if (i2 + 2 == i1):
        return True
    else:
        return False


def n(name):
    if isinstance(name, int):
        return 'module.conv' + str(name) + '.weight'
    else:
        return 'module.' + name + '.weight'


def buildResidualPath(module_list, numOfStages, archNums):
    # stage0O = [n(1), n(3), n(5), n(7), n(9), n(11)]
    # stages1O = [n(13), n(14), n(16), n(18), n(20), n(22)]
    # stages2O = [n(24), n(25), n(27), n(29), n(31), n(33)]
    printStages = False
    sameNode, oddLayers = buildShareSameNodeLayers(module_list, numOfStages, archNums)
    tempStagesI = []
    tempStagesO = [n(1)]
    stageWidth = module_list[0].weight.size()[0]
    oddLayersCopy = oddLayers
    oddLayersBool = False
    for node in sameNode:
        if len(oddLayers) > 0:
            # print(f'oddLayer: {self.oddLayers[0]}')
            if compare(node[-1], oddLayers[0]):
                oddLayer = oddLayers.pop(0)
                tempStagesO.append(oddLayer)
                tempStagesI.append(oddLayer)
                oddLayersBool = True
        tempStagesI.append(node[0])
        tempStagesO.append(node[-1])

    length = len(module_list)
    fcStr = 'fc' + str(int(length / 2))
    tempStagesI.append(n(fcStr))
    stagesI = [[]]
    stagesO = [[]]
    for layer in tempStagesI:
        # print(layer)
        if 'conv' in layer:
            i = int(layer.split('.')[1].split('v')[1])
            i = 2 * i - 2
            if i == 0:
                stagesI[0].append(layer)
            elif module_list[i].weight.size()[1] == stageWidth:
                stagesI[-1].append(layer)
            else:
                stageWidth = module_list[i].weight.size()[1]
                stagesI.append([])
                stagesI[-1].append(layer)

        elif 'fc' in layer:
            stagesI[-1].append(layer)
        # print(f'StagesI:{stagesI}')

    stageWidth = module_list[0].weight.size()[0]
    for layer in tempStagesO:
        # print(layer)
        i = int(layer.split('.')[1].split('v')[1])
        i = 2 * i - 2
        if module_list[i].weight.size()[0] == stageWidth:
            stagesO[-1].append(layer)
        elif layer in oddLayersCopy:
            stagesO[1].append(layer)
        else:
            stageWidth = module_list[i].weight.size()[0]
            stagesO.append([])
            stagesO[-1].append(layer)

    # print(f'stagesI: {stagesI}')

    # print(f'stagesO: {stagesO}')
    return stagesI, stagesO


def buildShareSameNodeLayers(module_list, numOfStages, archNums):
    sameNode = []
    oddLayers = []
    first = True
    j = 2
    k = 2
    firstStage = True
    for stage in range(0, numOfStages):
        firstBlockInStage = True
        for i in range(0, len(archNums[stage])):
            block = []
            for layer in range(0, archNums[stage][i]):
                # print("\nI: ", i, " ; ", stage, " ; ", block, " ; ", layer)
                if (layer + 1) % archNums[stage][i] == 0 and not firstStage and firstBlockInStage:
                    oddLayers.append(n(k))
                    j = j + 1
                    k = k + 1
                    firstBlockInStage = False
                if isinstance(module_list[j], nn.Conv2d):
                    block.append(n(k))
                    j = j + 1
                    k = k + 1
                if isinstance(module_list[j], nn.BatchNorm2d):
                    j = j + 1

            sameNode.append(block)
        firstStage = False
    # print(f'oddLayers: {oddLayers}')
    # print("\nSame Node: ", sameNode)
    return sameNode, oddLayers
