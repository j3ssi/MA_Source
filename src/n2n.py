import copy
import torch
import torch.nn as nn
import math


class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock, first, model=None):
        super(N2N, self).__init__()
        self.numOfStages = numOfStages
        self.numOfBlocksinStage = numOfBlocksinStage
        self.layersInBlock = layersInBlock
        if first:
            self.archNums =[[]]
            for s in range(0, self.numOfStages):
                # print("\nS: ", s, " ; ", self.numOfStages)
                for b in range(0, self.numOfBlocksinStage):
                    self.archNums[s].append(self.layersInBlock)
                if s != (self.numOfStages-1):
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
            firstLayer = True
            for stage in range(0, numOfStages):
                firstLayerInStage = True
                sizeOfLayer = pow(2, stage + 4)
                # print("\nStage: ", stage, " ; ", sizeOfLayer)
                for block in range(0, numOfBlocksinStage):
                    i = 0
                    while i < self.layersInBlock:
                        if firstLayerInStage and not firstLayer:
                            conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                             stride=1)
                            self.module_list.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            self.module_list.append(bn)
                            i = i + 1
                            firstLayerInStage = False

                        else:
                            conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False, stride=1)
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
            self.archNums = model.archNums

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

                elif 'bn' in name and not 'bias' in name:
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
        for stage in range(0, self.numOfStages):
            if printNet:
                print("\n\nStage: ", stage)
            archNum = self.archNums[stage]
            for block in range(0, len(archNum)):
                if printNet:
                    print("\n\n\tBlock: ", block)
                i = 0
                layerInThisBlock = archNum[block]
                while i < layerInThisBlock:
                    if i == 0:
                        # conv
                        x = self.module_list[j](_x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                        j = j + 1
                        i = i + 1

                        x = self.relu(x)

                    elif ((i + 1) % self.layersInBlock) == 0 and (block > 0 or stage == 0):
                        # conv
                        x = self.module_list[j](x)
                        if printNet:
                            print("\n J: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nShortcutLayer J: ", j, " ; ", self.module_list[j])
                        j = j + 1
                        i = i + 1

                        _x = _x + x
                        _x = self.relu(_x)


                    elif ((i + 1) % self.layersInBlock) == 0:

                        # conv
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1

                        # bn
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                        j = j + 1
                        i = i + 1
                        _x = self.relu(x)

                    else:
                        # conv
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1
                        # bn
                        x = self.module_list[j](x)
                        if printNet:
                            print("\nJ: ", j, " ; ", self.module_list[j])
                            print("\nX Shape: ", x.shape)
                        j = j + 1
                        x = self.relu(x)
                        i = i + 1

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
        else:
            print("\n \n Oops!!!: ")
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
        stagesI = []
        stagesO = []
        i = 2
        printStages = False
        stagesI.append([])
        stagesO.append([])
        stagesO[0].append(n(2))
        for stage in range(0, self.numOfStages):
            if stage < self.numOfStages:
                i = i + 1
                print(f'I: {i} ; {stage}')
                stagesI[-1].append(n(i))
            if stage > 0:
                stagesI.append([])
                stagesO.append([])
            if printStages:
                print("\n\nStage: ", stage)
            archNum = self.archNums[stage]
            for block in range(0, len(archNum)):
                if printStages:
                    print("\n\n\tBlock: ", block)
                if 0 < block < len(archNum):
                    i = i + 1
                    stagesI[-1].append(n(i))
                    layerInThisBlock = archNum[block]
                    i = i +layerInThisBlock - 1
                    stagesO[-1].append(n(i))
                elif block == 0:
                    layerInThisBlock = archNum[block]
                    i = i + layerInThisBlock -1
                    stagesO[-1].append(n(i))

        # print("\nstagesO:  1")
        printStages = False
        fcStr = 'fc' + str(i+1)
        stagesI[-1].append(n(fcStr))

        # if printStages:
        print("\nStagesI: ", stagesI)
        print("\nStagesO: ", stagesO)
        return stagesI, stagesO

    def getShareSameNodeLayers(self):
        sameNode = []
        first = True
        i = 2
        for stage in range(0, self.numOfStages):
            for block in range(0, self.numOfBlocksinStage):
                block = []
                for layer in range(0, self.layersInBlock):
                    # print("\nI: ", i, " ; ", stage, " ; ", block, " ; ", layer)
                    if (i - 1) % self.layersInBlock == 1:
                        block.append(n(i))
                        i = i + 1
                    elif (i - 1) % self.layersInBlock == 0:
                        block.append(n(i))
                        i = i + 1
                    else:
                        block.append(n(i))
                        i = i + 1
                sameNode.append(block)

        sameNode.append(block)
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
