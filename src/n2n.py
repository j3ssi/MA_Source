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
            # first Layer
            self.module_list = nn.ModuleList()
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
                                             stride=2)
                            self.module_list.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            self.module_list.append(bn)
                            i = i + 1
                            firstLayerInStage = False
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            self.module_list.append(bn)
                            i = i + 1
                        else:
                            conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False, stride=1)
                            self.module_list.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            self.module_list.append(bn)
                            i = i + 1
                    firstLayer = False
            # 18
            self.sizeOfFC = pow(2, self.numOfStages + 2)
            print("\n self sizeofFC: ",self.sizeOfFC)
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
            print(self)
        else:
            altList = []
            paramList = []
            printName = True
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
            # print("\naltList", altList)
            module_list1 = nn.ModuleList()
            for i in range(0, len(altList)):
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
                    print("\n>new Layer: ", layer, " ; ", param.shape)
                    print("\nWeight Shape: ", module.weight.shape)
                    layer.weight = module.weight
                    module_list1.append(layer)

                elif 'bn' in name and not 'bias' in name:
                    layer = nn.BatchNorm2d(paramList[i].shape[0])
                    print("\n>new Layer: ", layer)
                    module_list1.append(layer)
                elif 'bn' in name and 'bias' in name:
                    print("\n>Name: ", name, " ; ", k)
                    k = int(name.split('.')[1].split('n')[1])
                    k1 = 2 * (k - 1) + 1
                    # print("\nk1: ", k1)
                    module = model.module_list[k1]
                    module_list1[-1].bias = module.bias
                    module_list1[-1].weight = module.weight
                # else:
                # print('\nelse: ', name)

            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.module_list.append(avgpool)
            module = module_list[-1]
            self.sizeOfFC = module.weight.shape[1]
            # print("\n self sizeofFC: ", self.sizeOfFC)
            fc = nn.Linear(module.weight.shape[1], num_classes)
            # print("\nLinear: ", fc)
            fc.weight = module.weight
            fc.bias = module.bias
            self.module_list.append(fc)
            self.relu = nn.ReLU(inplace=True)
            # print("\nnew Model: ", self)

    def forward(self, x):
        # First layer
        printNet = True
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

            for block in range(0, self.numOfBlocksinStage):
                firstLayerInBlock = True
                if printNet:
                    print("\n\n\tBlock: ", block)
                i=0
                while i < self.layersInBlock:
                    if firstLayerInBlock:
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

                        firstLayerInBlock = False


                    elif i % (self.layersInBlock - 1):
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

                        _x = _x + x
                        _x = self.relu(_x)
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
                    firstLayerInBlock = False

        if isinstance(self.module_list[j], nn.AdaptiveAvgPool2d):
            try:
                x = self.module_list[j](_x)
                if printNet:
                    print("\nJ: ", j, " ; ", self.module_list[j])
                    print("\n\n X Shape 1: ", x.shape)
                x = x.view(-1, self.sizeOfFC)
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

    # 2 -> 2    0 -> 2
    # 4 -> 3    2 -> 1
    # 6 -> 4    4 -> 2
    # 8 -> 5    6 -> 3
    def getResidualPath(self):
        stagesI = []
        stagesO = []
        i = 1
        stagesI.append([])
        stagesO.append([])
        stagesO[0].append(n(1))
        firstBlock = False
        printStages = True
        for stage in range(0, self.numOfStages):
            for block in range(0, self.numOfBlocksinStage):
                for layer in range(0, self.layersInBlock):
                    if not firstBlock:
                        if (i - 1) % self.layersInBlock == 0:
                            stagesI[stage].append(n(int(i - 2 / 2)))
                            i = i + 1
                            if printStages:
                                print("\nI: ", i)
                        if (i - 1) % self.layersInBlock == self.layersInBlock - 1:
                            stagesO[stage].append(n(int(i - 2 / 2)))
                            i = i + 1
                            if printStages:
                                print("\nI: ", i)
                        else:
                            i = i + 1
                    else:
                        if (i - 1) % self.layersInBlock == self.layersInBlock - 1:
                            stagesO[stage].append(n(int(i - 2 / 2)))
                            i = i + 1
                            firstBlock = False
                            if printStages:
                                print("\nI: ", i)

                        else:
                            i = i + 1
            if self.layersInBlock > 2:
                stagesI[stage].append(n(int(i - 2 / 2)))
                i = i + 1
                stagesI[stage].append(n(int(i - 2 / 2)))
                i = i + 1
            else:
                stagesI[stage].append(n(int(i + 2 / 2)))
                print("\nI: ", i)
                i = i + 1

        stageStr = 'fc' + str(i + 1)
        stagesI[-1].append(n(stageStr))
        print("\nStagesI: ", stagesI)
        print("\nStagesO: ", stagesO)
        return stagesI, stagesO

    def getShareSameNodeLayers(self):
        sameNode = []
        first = True
        i = 2
        for stage in range(0, self.numOfStages):
            for block in range(0, self.numOfBlocksinStage):
                for layer in range(0, self.layersInBlock):
                    # print("\nI: ", i, " ; ", stage, " ; ", block, " ; ", layer)
                    if i % 2 == 0:
                        sameNode.append((n(i), n(i + 1)))
                        i = i + 1
                    else:
                        i = i + 1

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
