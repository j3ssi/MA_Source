import copy

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
        self.device = torch.device("cuda:0")
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
            for stage in numOfStages:
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
            # print(self)
            self.sameNode, self.oddLayers = buildShareSameNodeLayers(self.module_list, self.numOfStages, self.archNums)
            self.stageI, self.stageO = buildResidualPath(self.module_list, self.numOfStages, self.archNums)
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
            # if printName:
            # print("\nnew Model: ", self)

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
                        print("\n\n\tBlock: ", block)
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
                                print("\ni: ", i, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            _x = self.module_list[j](_x)
                            if printNet:
                                print("\ni: ", i, " ; ", self.module_list[j])
                            j = j + 1
                            i = i + 1
                            firstBlockInStage = False
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
        print(self)
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
        # print(f'Model before wider: {self}')
        altList = []
        paramList = []
        printName = False
        for name, param in self.named_parameters():
            # print("\nName: {}", name)
            paramList.append(param)
            # print("\nName: ", name)
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
                    if tobestage ==stage:
                        num = int(altList[index].split('.')[1].split('v')[1])
                        residualListO.append(num)
            if 'fc' in altList[index]:
                stage = self.widthofLayers[-1]
                num = int(altList[index].split('.')[1].split('c')[1])
                print(f'num: {num}')

        sameNodes = self.getShareSameNodeLayers()
        tmpListI = copy.copy(residualListI)
        tmpListO = copy.copy(residualListO)
        residualList = sorted(tmpListI | tmpListO)
        print(f'residualI: {residualListI}')
        print(f'residualO: {residualListO}')
        index = 0
        while index == 0:
            j = residualList.pop(0)
            if int(sameNodes[0][0].split('.')[1].split('v')[1]) == j:
                sameNode = sameNodes.pop(0)
                sameNode.pop(0)
                mapNodes = map(lambda x: int(x.split('.')[1].split('v')[1]), sameNode)
                residualList = sorted(list(set(residualList) | set(mapNodes)))
                # print(f'sameNodes: {sameNode}')

            print(f'list: {list(residualList)}; j: {j}')
            i = 2 * j - 2
            m1 = self.module_list[i]
            bn = self.module_list[i + 1]
            print(f'm1: {m1}')
            print(f'bn1: {bn}')
            w1 = m1.weight.data
            w1list = m1.weight.data.cpu().numpy().tolist()

            # Fehlersuche
            assert delta_width > 0, "New size should be larger"

            print(f'dim w1: {w1.size()}')

            layerinbetween = False

            if w1.size(0) == w1.size(1):
                layerinbetween = True
            if j in residualListI or layerinbetween:
                print(f'maplistI: {list(residualListI)}; layerin: {layerinbetween}')
                print(f'in maplistI j: {j}')
                old_width = w1.size(1)
                new_width = old_width * delta_width
                print(f'old width: {old_width}')
                dw1 = []
                tracking = dict()
                listOfNumbers = []
                listindices = []
                for o in range(0, (new_width - old_width)):
                    idx = np.random.randint(0, old_width)
                    m1list = m1.weight[:, idx, :, :].data.cpu().numpy().tolist()
                    listindices.append(idx)
                    # print(f'listindices beim befüllen: {listindices}')
                    # print(f'm1list: {m1list}')
                    # print(f'm2list: {m2list}')
                    # print(f'idx: {idx}')
                    try:
                        tracking[idx].append(o)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o)

                    # TEST:random init for new units
                    if random_init:
                        n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        # if m2.weight.dim() == 4:
                        #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                        # elif m2.weight.dim() == 5:
                        #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                        # elif m2.weight.dim() == 2:
                        #    n2 = m2.out_features * m2.in_features
                        # dw1.select(0, i).normal_(0, np.sqrt(2. / n))
                        # dw2.select(0, i).normal_(0, np.sqrt(2. / n2))
                    else:
                        dw1.append(m1list)
                        # dw2.append(m2list)
                        # dw1.select(0, i).copy_(w1.select(0, idx).clone())
                        # dw2.select(0, i).copy_(w2.select(0, idx).clone())

                # print(f'indices: {listindices}')
                # print(f'tracking dict: {tracking}')
                ct = {}
                for key, dif_k in tracking.items():
                    # print(f'key: {key}; difk: {dif_k}')
                    dictcounter = len(dif_k)
                    ct.update({key: dictcounter})
                # print(f'ct: {ct}')

                # print(f'len: {len(listOfRunningMean)}')
                # print(f'm1list: {m1list}')

                w11 = torch.FloatTensor(dw1).transpose(0, 1).cuda()
                print(f'dim w1: {w1.size()}; dim w11: {w11.size()}')

                w1 = torch.cat((w1, w11), dim=1)
                print(f'dim w1: {w1.size()}')
                m1.in_channels = new_width
                i0 = w1.size()[0]
                i1 = w1.size()[1]
                i2 = w1.size()[2]
                i3 = w1.size()[3]
                x = w1.std()
                print(f'i0: {i0}; i1: {i1}; i2: {i2}; i3: {i3}')
                if addNoise:
                    noise = np.random.normal(scale=5e-2 * 0.3,
                                             size=(i0, i1, i2, i3))
                    w1 += th.FloatTensor(noise).type_as(w1)

                m1.weight.data = w1

            if j in residualListO or layerinbetween:
                print(f'in maplistO j: {j}')

                old_width = w1.size(0)
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
                bn1list = bn.bias.data.cpu().numpy().tolist()
                bn1wlist = bn.weight.data.cpu().numpy().tolist()
                for name, buf in self.named_buffers():
                    # print("\nBuffer Name: ", name)
                    if 'running_mean' in name:
                        k = int(name.split('.')[1])
                        if (k == (i + 1)):
                            listOfRunningMean = buf.cpu().numpy().tolist()

                    if 'running_var' in name:
                        k = int(name.split('.')[1])
                        if (k == (i + 1)):
                            listOfRunningVar = buf.cpu().numpy().tolist()

                listindices = []
                print(f'oldwidth: {old_width} ')
                for o in range(0, (new_width - old_width)):
                    idx = np.random.randint(0, old_width)
                    m1list = m1.weight[idx, :, :, :].data.cpu().numpy().tolist()
                    listindices.append(idx)
                    # print(f'listindices beim befüllen: {listindices}')
                    # print(f'm1list: {m1list}')
                    print(f'idx: {idx}')
                    try:
                        tracking[idx].append(o)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o)

                    # TEST:random init for new units
                    if random_init:
                        n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        # if m2.weight.dim() == 4:
                        #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                        # elif m2.weight.dim() == 5:
                        #    n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                        # elif m2.weight.dim() == 2:
                        #    n2 = m2.out_features * m2.in_features
                        # dw1.select(0, i).normal_(0, np.sqrt(2. / n))
                        # dw2.select(0, i).normal_(0, np.sqrt(2. / n2))
                    else:
                        dw1.append(m1list)
                        # dw1.select(0, i).copy_(w1.select(0, idx).clone())
                        # dw2.select(0, i).copy_(w2.select(0, idx).clone())

                    if bn is not None:
                        # print(f'listofRunning mean: {listOfRunningMean.pop(0)}')
                        dbn1 = listOfRunningMean[idx]
                        # print(f'length of dbn1: {dbn1}')
                        dbn1rm.append(dbn1)
                        dbn1 = listOfRunningVar[idx]
                        dbn1rv.append(dbn1)
                        # print(f'running mean: {dbn1rm}')
                        # dbn1rm[i] = bn.running_mean.data()[idx]
                        # dbn1rv[i] = bn.running_var.data()[idx]
                        if bn.affine:
                            dbn1w.append(bn1wlist[idx])
                            dbn1b.append(bn1list[idx])
                    bn.num_features = new_width
                print(f'indices: {listindices}')
                print(f'tracking dict: {tracking}')
                ct = {}
                for key, dif_k in tracking.items():
                    print(f'key: {key}; difk: {dif_k}')
                    dictcounter = len(dif_k)
                    ct.update({key: dictcounter})
                print(f'ct: {ct}')
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

                # print(f'len: {len(listOfRunningMean)}')
                # print(f'm1list: {m1list}')

                w11 = torch.FloatTensor(dw1).cuda()
                print(f'dim w1: {w1.size()}; dim w11: {w11.size()}')

                w1 = torch.cat((w1, w11), dim=0)
                print(f'dim w1: {w1.size()}')
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
                # i0 = len(w1list) * delta_width
                # i1 = len(dw1[0])
                # i2 = len(dw1[0][0])
                # i3 = len(dw1[0][0][0])
                # x = w1.std()
                # print(f'i0: {i0}')
                # if noise:
                #     noise = np.random.normal(scale=5e-2 * 0.3,
                #                              size=(i0, i1, i2, i3))
                #     w1 += th.FloatTensor(noise).type_as(w1)

                m1.weight.data = w1
                if bn is not None:
                    bn.running_var = nbn1rv
                    bn.running_mean = nbn1rv
                    if bn.affine:
                        bn.weight.data = nbn1w
                        bn.bias.data = nbn1b

            if len(residualList) == 0:
                index = 1
            # print(f'Model after wider: {self}')
        # def deeper(self, model, optimizer):
        #     # each pos in pisitions is the position in which the layer sholud be duplicated to make the cnn deeper
        #     # for stage in self.archNums[i]:
        #         # print("\n\nposition:")
        #         # print(pos)
        #     conv = model.module_list[pos * 2 - 2]
        #     bn = model.module_list[pos * 2 - 1]
        #     conv1 = model.module_list[pos * 2]
        #     bn1 = model.module_list[pos * 2 + 1]
        #     conv2 = copy.deepcopy(conv)
        #     conv3 = copy.deepcopy(conv1)
        #     noise = torch.Tensor(conv2.weight.shape).random_(0, 1).to(self.device)
        #     # noise = torch.rand(0,0.5)
        #     conv2.weight.data += noise
        #     bn2 = copy.deepcopy(bn)
        #     noise = torch.Tensor(conv1.weight.shape).random_(0, 1).to(self.device)
        #     conv3.weight.data += noise
        #     bn3 = copy.deepcopy(bn1)
        #     model.module_list.insert(pos * 2 + 2, conv2)
        #     model.module_list.insert(pos * 2 + 3, bn2)
        #     model.module_list.insert(pos * 2 + 4, conv3)
        #     model.module_list.insert(pos * 2 + 5, bn3)

        # return model


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
