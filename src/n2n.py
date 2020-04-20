import copy
import torch as th
import torch.nn as nn
import math
import numpy as np




class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock,
                 first, bottleneck, model=None, archNums=None):
        super(N2N, self).__init__()
        self.numOfStages = numOfStages
        self.oddLayers = []
        self.numOfBlocksinStage = numOfBlocksinStage
        self.bottleneck = bottleneck
        self.layersInBlock = layersInBlock
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
            conv0 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv0)
            # bn1
            bn1 = nn.BatchNorm2d(16)
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
                    sizeOfLayer = pow(2, stage + 4)
                    # print("\nStage: ", stage, " ; ", sizeOfLayer)
                    for block in range(0, len(self.archNums[stage])):
                        i = 0
                        while i < self.archNums[stage][block]:
                            # print(f'i : {i}')
                            if firstBlockInStage and not firstLayer and i == 0:
                                conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                                 bias=False,
                                                 stride=2)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                i = i + 1

                            elif firstBlockInStage and not firstLayer and (i + 1) % self.archNums[stage][block] == 0:

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
                self.sizeOfFC = pow(2, self.numOfStages + 3)

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
            # print(self)
        else:
            self.archNums = archNums
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

            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.module_list.append(avgpool)
            module = module_list[-1]
            self.sizeOfFC = module.weight.shape[1]
            if printName:
                print("\n self sizeofFC: ", self.sizeOfFC)
            fc = nn.Linear(module.weight.shape[1], num_classes)
            if printName:
                print("\nLinear: ", fc)
            fc.weight.data = module.weight.data
            fc.bias.data = module.bias.data
            self.module_list.append(fc)
            self.relu = nn.ReLU(inplace=True)
            # if printName:
            print("\nnew Model: ", self)

    def forward(self, x):
        # First layer
        printNet = True
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

    def getResidualPath(self):
        # stage0O = [n(1), n(3), n(5), n(7), n(9), n(11)]
        # stages1O = [n(13), n(14), n(16), n(18), n(20), n(22)]
        # stages2O = [n(24), n(25), n(27), n(29), n(31), n(33)]
        printStages = False
        sameNode = self.getShareSameNodeLayers()
        tempStagesI = []
        tempStagesO = [n(1)]
        stageWidth = self.module_list[0].weight.size()[0]
        oddLayersBool = False
        for node in sameNode:
            if len(self.oddLayers) > 0:
                # print(f'oddLayer: {self.oddLayers[0]}')
                if compare(node[-1], self.oddLayers[0]):
                    oddLayer = self.oddLayers.pop(0)
                    tempStagesO.append(oddLayer)
                    oddLayersBool =True
            tempStagesI.append(node[0])
            tempStagesO.append(node[-1])

        length = len(self.module_list)
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
                elif self.module_list[i].weight.size()[1] == stageWidth:
                    stagesI[-1].append(layer)
                else:
                    stageWidth = self.module_list[i].weight.size()[1]
                    stagesI.append([])
                    stagesI[-1].append(layer)

            elif 'fc' in layer:
                stagesI[-1].append(layer)
            # print(f'StagesI:{stagesI}')

        stageWidth = self.module_list[0].weight.size()[0]
        for layer in tempStagesO:
            # print(layer)
            i = int(layer.split('.')[1].split('v')[1])
            i = 2 * i - 2
            if self.module_list[i].weight.size()[0] == stageWidth:
                stagesO[-1].append(layer)
            else:
                stageWidth = self.module_list[i].weight.size()[1]
                stagesO.append([])
                stagesO[-1].append(layer)

        print(f'stagesI: {stagesI}')

        print(f'stagesO: {stagesO}')
        return stagesI, stagesO


    def getShareSameNodeLayers(self):
        sameNode = []
        self.oddLayers = []
        first = True
        j = 2
        k = 2
        firstStage = True
        for stage in range(0, self.numOfStages):
            firstBlockInStage = True
            for i in range(0, len(self.archNums[stage])):
                block = []
                for layer in range(0, self.archNums[stage][i]):
                    # print("\nI: ", i, " ; ", stage, " ; ", block, " ; ", layer)
                    if (layer + 1) % self.archNums[stage][i] == 0 and not firstStage and firstBlockInStage:
                        self.oddLayers.append(n(k))
                        j = j + 1
                        k = k + 1
                        firstBlockInStage = False
                    if isinstance(self.module_list[j], nn.Conv2d):
                        block.append(n(k))
                        j = j + 1
                        k = k + 1
                    if isinstance(self.module_list[j], nn.BatchNorm2d):
                        j = j + 1

                sameNode.append(block)
            firstStage = False
        # print(f'oddLayers: {self.oddLayers}')
        print("\nSame Node: ", sameNode)
        return sameNode

    def delete(self, model, index):
        printNet = True
        print(f'Index1: {index}')
        index = int(index / 2 + 1)
        print(f'Index: {index}')
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
                    print(f'j: {j}; k: {k}')
                    if j == index:
                        numDelete = self.archNums[stage][block]
                        stageDelete = stage
                        blockDelete = block
                        print(f'numDelete: {numDelete}')
                    j = j + 1
                    i = i + 1

        module_list = nn.ModuleList()
        deleteModule = True
        for layers in range(0, (len(self.module_list) - 2 * numDelete)):
            if layers < (2 * k - 2):
                module_list.append(self.module_list[layers])
                print(f'Kopiere {layers}: {module_list[layers]}')
            elif layers - 2 * numDelete < (2 * k - 2):
                if isinstance(self.module_list[layers], nn.Conv2d) and isinstance(
                        self.module_list[layers + 2 * numDelete], nn.Conv2d):
                    print(f'Shape1: {self.module_list[layers].weight.size()}')
                    print(f'Shape2: {self.module_list[layers + 2 * numDelete].weight.size()}')
                    inChannels1 = self.module_list[layers].weight.size()[1]
                    inChannels2 = self.module_list[layers + layers + 2 * numDelete].weight.size()[1]
                    outChannels1 = self.module_list[layers].weight.size()[0]
                    outChannels2 = self.module_list[layers + layers + 2 * numDelete].weight.size()[0]
                    if not (inChannels1 == inChannels2) and i == 0:
                        print(f'InChannels haben nicht die gleiche Dimension')
                        deleteModule = False
                        break
                    if not (outChannels1 == outChannels2) and i == (numDelete * 2 - 1):
                        print(f'InChannels haben nicht die gleiche Dimension')
                        deleteModule = False
                        break
                    module_list.append(self.module_list[layers + 2 * numDelete])
                    print(
                        f'Ersetze {layers} gegen {layers + 2 * numDelete}: {self.module_list[layers]} gegen {self.module_list[layers + 2 * numDelete]}')
                else:
                    module_list.append(self.module_list[layers + 2 * numDelete])
                    print(
                        f'Ersetze {layers} gegen {layers + 2 * numDelete}: {self.module_list[layers]} gegen {self.module_list[layers + 2 * numDelete]}')
            elif layers < len(self.module_list) - 2 * numDelete:
                if isinstance(self.module_list[layers], nn.Conv2d):
                    print(f'Shape1: {self.module_list[layers].weight.size()}')
                    if isinstance(self.module_list[layers + 2 * numDelete - 1], nn.AdaptiveAvgPool2d):
                        print(f'Shape2: {self.module_list[layers + 2 * numDelete - 1].weight.size()}')
                    elif layers in blockBegin:
                        inChannels1 = self.module_list[layers].weight.size()[1]
                        inChannels2 = self.module_list[layers + layers + 2 * numDelete - 1].weight.size()[1]
                        if not (inChannels1 == inChannels2):
                            print(f'InChannels haben nicht die gleiche Dimension')
                            deleteModule = False
                            break
                    elif (layers + 1) in blockBegin:
                        outChannels1 = self.module_list[layers].weight.size()[0]
                        outChannels2 = self.module_list[layers + layers + 2 * numDelete - 1].weight.size()[0]
                        if not (outChannels1 == outChannels2):
                            print(f'InChannels haben nicht die gleiche Dimension')
                            deleteModule = False
                            break
                    module_list.append(self.module_list[layers + 2 * numDelete])
                    print(
                        f'Ersetze {layers} gegen {layers + 2 * numDelete}: {self.module_list[layers]} gegen {self.module_list[layers + 2 * numDelete]}')
                else:
                    module_list.append(self.module_list[layers + 2 * numDelete])
                    print(f'Ersetze {layers} gegen {layers + 2 * numDelete}: {self.module_list[layers]}')

        if deleteModule:
            self.archNums[stage][block] = 0
            self.archNums[stage].remove(0)
            self.module_list = module_list
        print(self)
        return model

    """
    Convert all layers in layer to its wider version by adapting next weight layer and possible batch norm layer in btw.
    layers = 'conv 3, conv6'
    """

    def wider(self, model, layers, delta_width, out_size=None, weight_norm=True, random_init=True, noise=True):
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
        j = 0
        residualPathI, residualPathO = model.getResidualPath()
        sameNodes = model.getShareSameNodeLayers()
        for layer in layers:
            if layer in residualPathO:
                # Do nothing
                continue
            else:
                j = int(name.split('.')[1].split('v')[1])
                i = 2 * j - 2
                m1 = model.module_list[j]
                bn = model.module_list[j + 1]
                m2 = model.module_list[j + 2]
                w1 = m1.weight.data
                w2 = m2.weight.data
                if w1.dim() == 4:
                    factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                    w2 = w2.view(w2.size(0), w2.size(1) // factor ** 2, factor, factor)
                elif w1.dim() == 5:
                    assert out_size is not None, \
                        "For conv3d -> linear out_size is necessary"
                    factor = out_size[0] * out_size[1] * out_size[2]
                    w2 = w2.view(w2.size(0), w2.size(1) // factor, out_size[0],
                                 out_size[1], out_size[2])
                assert delta_width > 0, "New size should be larger"

                old_width = w1.size(0)
                nw1 = w1.clone()
                nw2 = w2.clone()

                if nw1.dim() == 4:
                    nw1.resize_(nw1.size(0) + delta_width, nw1.size(1), nw1.size(2), nw1.size(3))
                    nw2.resize_(nw2.size(0), nw1.size(0) + delta_width, nw2.size(2), nw2.size(3))
                elif nw1.dim() == 5:
                    nw1.resize_(nw1.size(0) + delta_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
                    nw2.resize_(nw2.size(0), nw1.size(0) + delta_width, nw2.size(2), nw2.size(3), nw2.size(4))
                else:
                    nw1.resize_(nw1.size(0) + delta_width, nw1.size(1))
                    nw2.resize_(nw2.size(0), nw1.size(0) + delta_width)

                nrunning_mean = bn.running_mean.clone().resize(nw1.size(0) + delta_width)
                nrunning_var = bn.running_var.clone().resize_(nw1.size(0) + delta_width)
                if bn.affine:
                    nweight = bn.data.clone().resize_(nw1.size(0) + delta_width)
                    nbias = bn.bias.data.clone().resize_(nw1.size(0) + delta_width)

                w2 = w2.transpose(0, 1)
                nw2 = nw2.transpose(0, 1)

                nw1.narrow(0, 0, old_width).copy_(w1)
                nw2.narrow(0, 0, old_width).copy_(w2)
                if bn is not None:
                    nrunning_var.narrow(0, 0, old_width).copy_(bn.running_var)
                    nrunning_mean.narrow(0, 0, old_width).copy_(bn.running_mean)
                    if bn.affine:
                        nweight.narrow(0, 0, old_width).copy_(bn.weight.data)
                        nbias.narrow(0, 0, old_width).copy_(bn.bias.data)

                # TEST:normalize weights
                if weight_norm:
                    for i in range(old_width):
                        norm = w1.select(0, i).norm()
                        w1.select(0, i).div_(norm)

                # select weights randomly
                tracking = dict()
                for i in range(old_width, nw1.size(0) + delta_width):
                    idx = np.random.randint(0, old_width)
                    try:
                        tracking[idx].append(i)
                    except:
                        tracking[idx] = [idx]
                        tracking[idx].append(i)

                    # TEST:random init for new units
                    if random_init:
                        n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        if m2.weight.dim() == 4:
                            n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
                        elif m2.weight.dim() == 5:
                            n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
                        elif m2.weight.dim() == 2:
                            n2 = m2.out_features * m2.in_features
                        nw1.select(0, i).normal_(0, np.sqrt(2. / n))
                        nw2.select(0, i).normal_(0, np.sqrt(2. / n2))
                    else:
                        nw1.select(0, i).copy_(w1.select(0, idx).clone())
                        nw2.select(0, i).copy_(w2.select(0, idx).clone())

                if bn is not None:
                    nrunning_mean[i] = bn.running_mean[idx]
                    nrunning_var[i] = bn.running_var[idx]
                    if bn.affine:
                        nweight[i] = bn.weight.data[idx]
                        nbias[i] = bn.bias.data[idx]
                    bn.num_features = nw1.size(0) + delta_width

                if not random_init:
                    for idx, d in tracking.items():
                        for item in d:
                            nw2[item].div_(len(d))

                w2.transpose_(0, 1)
                nw2.transpose_(0, 1)

                m1.out_channels = nw1.size(0) + delta_width
                m2.in_channels = nw1.size(0) + delta_width

                if noise:
                    noise = np.random.normal(scale=5e-2 * nw1.std(),
                                             size=list(nw1.size()))
                    nw1 += th.FloatTensor(noise).type_as(nw1)

                m1.weight.data = nw1
                m2.weight.data = nw2

                if bn is not None:
                    bn.running_var = nrunning_var
                    bn.running_mean = nrunning_mean
                    if bn.affine:
                        bn.weight.data = nweight
                        bn.bias.data = nbias
        return model

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
        #     noise = torch.Tensor(conv2.weight.shape).random_(0, 1).cuda()
        #     # noise = torch.rand(0,0.5)
        #     conv2.weight.data += noise
        #     bn2 = copy.deepcopy(bn)
        #     noise = torch.Tensor(conv1.weight.shape).random_(0, 1).cuda()
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
    if(i2+2==i1):
        return True
    else:
        return False
def n(name):
    if isinstance(name, int):
        return 'module.conv' + str(name) + '.weight'
    else:
        return 'module.' + name + '.weight'
