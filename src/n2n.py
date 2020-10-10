import copy
import scipy.signal

import gitignore.Net2Net as n
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
        self.numOfStages = numOfStages
        self.oddLayers = []
        self.numOfBlocksinStage = numOfBlocksinStage
        self.bottleneck = bottleneck
        self.layersInBlock = layersInBlock
        if widthOfLayers is not None:
            self.widthofFirstLayer = widthOfLayers[0]
            self.widthofLayers = widthOfLayers
            print(f'width: {self.widthofFirstLayer}')
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
            print("\nArch Num: ", self.archNums)

            self.module_list = nn.ModuleList()

            # first Layer
            # conv1
            conv0 = nn.Conv2d(3, self.widthofFirstLayer, kernel_size=3, padding=1, bias=False, stride=1)
            self.module_list.append(conv0)
            # bn1
            bn1 = nn.BatchNorm2d(self.widthofFirstLayer)
            self.module_list.append(bn1)
            # print(f'ohne Bottleneck!')
            firstBlockInStage = False
            for stage in range(0, numOfStages):
                if self.widthofLayers is None:
                    sizeOfLayer = pow(2, stage)
                else:
                    sizeOfLayer = widthOfLayers[stage]
                # print(f'stage: {stage}; sizeof Layers: {sizeOfLayer}')
                # print("\nStage: ", stage, " ; ", sizeOfLayer)
                for block in range(0, len(self.archNums[stage])):
                    layer = []
                    layer2 = []
                    i = 0

                    while i < self.archNums[stage][block]:
                        print(f'i : {i}; block: {block}')
                        if firstBlockInStage and i == 0:
                            conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                             bias=False,
                                             stride=2)
                            print(f'{conv}')
                            layer.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            print(f'{bn}')
                            layer.append(bn)
                            i = i + 1

                        elif firstBlockInStage and (i + 1) % self.archNums[stage][block] == 0:
                            conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                             bias=False,
                                             stride=2)
                            print(f'{conv}')
                            layer2.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            print(f'{bn}')
                            layer2.append(bn)
                            i = i + 1
                            firstBlockInStage = False

                        elif firstBlockInStage:
                            conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1,
                                             bias=False,
                                             stride=1)
                            print(f'{conv}')
                            layer.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            # print(f'{bn}')
                            layer.append(bn)
                            i = i + 1

                        else:
                            conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                             stride=1)
                            print(f'{conv}')
                            layer.append(conv)
                            bn = nn.BatchNorm2d(sizeOfLayer)
                            print(f'{bn}')
                            layer.append(bn)
                            i = i + 1

                    block = nn.Sequential(*layer)
                    self.module_list.append(block)
                    if len(layer2) > 0:
                        block = nn.Sequential(*layer)
                        self.module_list.append((block))

                    layer2 = []
                    # 18

                firstBlockInStage = True

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
            # self.sameNode, self.oddLayers = buildShareSameNodeLayers(self.module_list, self.numOfStages, self.archNums)
            # self.stageI, self.stageO = buildResidualPath(self.module_list, self.numOfStages, self.archNums)
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
                    print("Name, Dims: ", name, " ; ", dims)
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
                if printNet:
                    print(f'Block: {block}; j: {j}')
                i = 0
                layerInThisBlock = archNum[block]
                seq = self.module_list[j]
                if block == 0 and stage > 0:
                    x = seq(_x)
                    j += 1
                    seq = self.module_list[j]
                    _x = seq(_x)
                    j += 1
                else:
                    x = seq(_x)
                    j += 1
                _x = x + _x
                _x = self.relu(_x)
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
        print(f'width of Layers: {self.widthofLayers}')
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
                width = paramList[index].size()[0]
                print(f'width: {width}')
                if self.widthofLayers.count(width) > 0:
                    tobestage = self.widthofLayers.index(width) + 1
                    print(f'stage: {stage}; tobestage: {tobestage}')

                    if tobestage == stage:
                        num = int(altList[index].split('.')[1].split('v')[1])
                        residualListI.append(num)
                        print(f'Num: {num}')
                width = paramList[index].size()[0]
                if self.widthofLayers.count(width) > 0:
                    tobestage = self.widthofLayers.index(width) + 1
                    # print(f'stage: {stage}')
                    if tobestage == stage:
                        num = int(altList[index].split('.')[1].split('v')[1])
                        residualListO.append(num)

        print(f'altList: {altList}')
        print(f'Residual ListI: {residualListI}')
        print(f'Residual ListO: {residualListO}')
        tmpListI = copy.copy(residualListI)
        tmpListO = copy.copy(residualListO)
        residualList = sorted(list(set(tmpListI) | set(tmpListO)))

        # fill numpy array with random elemente from original weight
        index = 0
        while index == 0:
            # get next elemente to widen
            j = residualList.pop(0)
            # transform to numbetr in moduleList
            if (j == 0):
                print(f'j')
                continue
            print(f'j: {j}')
            i = 2 * j - 2

            # get modules
            m1 = self.module_list[i]
            w1 = m1.weight.data.clone().cpu().numpy()
            bn = self.module_list[i + 1]
            bnw1 = bn.weight.data.clone().cpu().numpy()
            bnb1 = bn.bias.data.clone().cpu().numpy()
            assert delta_width > 0, "New size should be larger"

            if j in residualListI and not j == 1:
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
                        tracking[idx].append(o + old_width)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o + old_width)
                    # TEST:random init for new units
                    if random_init:
                        n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n),
                                                  size=(w1.shape[0], new_width - old_width, w1.shape[2], w1.shape[3]))
                        print(f'dw1: {dw1.shape}')
                    else:
                        dw1.append(m1list)
                ct = {}
                tracking_inverse = {}
                print(f'tracking items: {tracking}')
                for key, dif_k in tracking.items():
                    # print(f'key: {key}; difk: {dif_k}')
                    dictcounter = len(dif_k)
                    ct.update({key: dictcounter})
                    # print(f'dif key: {dif_k}')
                    for item in dif_k:
                        if item not in tracking_inverse:
                            tracking_inverse[item] = key
                        else:
                            tracking_inverse[item].append(key)
                print(f'ct: {ct}')
                print(f'invers: {tracking_inverse}')
                if not random_init:
                    for idx in range(0, (new_width - old_width)):
                        print(f'idx: {idx}')
                        c = dw1[idx]
                        x = tracking_inverse[idx + old_width]
                        y = int(ct[x])
                        print(f'tracking inverse[{idx + old_width}]: {tracking_inverse[idx + new_width - old_width]} ')
                        # print(f'c:{c}')
                        for k in range(len(c)):
                            e = c[k]
                            # print(f'c[k]: {c[k]}')
                            for l in range(len(e)):
                                # print(f' before e[l]: {e[l]}')
                                f = e[l]
                                for m in range(len(f)):
                                    f[m] = f[m] / y
                #                 print(f' after e[l]: {e[l]}')

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

                # print(f'oldwidth: {old_width} ')
                for o in range(0, (new_width - old_width)):
                    idx = np.random.randint(0, old_width)
                    m1list = w1[idx, :, :, :]
                    try:
                        tracking[idx].append(o + old_width)
                    except:
                        tracking[idx] = []
                        tracking[idx].append(o + old_width)

                    # TEST:random init for new units
                    if random_init:
                        n1 = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
                        dw1 = numpy.random.normal(loc=0, scale=np.sqrt(2. / n1),
                                                  size=(new_width - old_width, w1.shape[1], w1.shape[2], w1.shape[3]))
                        # print(f'dw1: {dw1.shape}')
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
                break
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
                else:
                    dw1.append(m1list)

            if not random_init:
                for idx in range(0, (new_width - old_width)):
                    print(f'idx: {idx}')
                    c = dw1x[idx]
                    x = tracking_inverse[idx + old_width]
                    y = int(ct[x])
                    print(f'tracking inverse[{idx + old_width}]: {tracking_inverse[idx + new_width - old_width]} ')
                    # print(f'c:{c}')
                    for k in range(len(c)):
                        c[k] = c[k] / y
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

    def deeper(self, pos=1):
        # make each block with plus two layers (conv +batch) deeper
        printDeeper = False
        newModule_list = nn.ModuleList()

        newModule_list.append(self.module_list[0])
        newModule_list.append(self.module_list[1])
        blockComp = False
        add = 0
        for i in range(len(self.module_list)):
            if i>3 and blockComp:
                blockComp = False
                add = add - 1
                continue
            if isinstance(self.module_list[i], nn.Sequential):
                # print(f'davor: {self.module_list[i]}')
                module = self.module_list[i]
                i0 = module[0].weight.size(0)
                i1 = module[0].weight.size(1)
                i2 = module[0].weight.size(2)
                i3 = module[0].weight.size(3)
                seq = []
                for j in range(len(module) + 2):
                    if j == 2 * pos - 1:
                        # print(f'Module {self.module_list[i]}; i: {i}')
                        bn = nn.BatchNorm2d(i0, eps=0)
                        torch.nn.init.ones_(bn.weight)
                        torch.nn.init.zeros_(bn.bias)
                        bn.running_mean.fill_(0)
                        bn.running_var.fill_(1)
                        seq.append(bn)
                        print(f'neues bn: {bn}; j: {j}')
                    if j == 2 * pos:
                        kernel_size = i2
                        stride = 1
                        padding = 1
                        conv = nn.Conv2d(i0, i0, kernel_size=kernel_size, stride=stride, padding=padding)
                        print(f'neues conv: {conv}; j: {j}')

                        # m = module[2 * pos - 2 ]
                        # deeper_w = np.zeros((i0, i0, i2, i3))
                        # deeper_w = torch.from_numpy(deeper_w)
                        # torch.nn.init.normal_(deeper_w, mean=0, std=0.5)
                        # deeper_w = deeper_w.numpy()
                        # center_h = ( i0 - 1) // 2
                        # center_w = ( i0 - 1) // 2
                        # for k in range( i3 ):
                        #     tmp = np.zeros(( i0, i0, i3))
                        #     tmp[center_h, center_w, k] = 1
                        #     deeper_w[:, :, :, k] = tmp
                        #     deeper_w = deeper_w.astype('float32')
                        # conv.weight.data = torch.from_numpy(deeper_w)

                        # for k in range(m.out_channels):
                        #     weight = m.weight.data
                        #     norm = weight.select(0, k).norm()
                        #     weight.div_(norm)
                        #     m.weight.data = weight
                        seq.append(conv)
                        # print(f'module: {conv}; j= { 2 * pos +1 }')
                    elif j > 2 * pos:
                        # print(f'module: {module[j - 2]}; j= {j + 2}')
                        print(f'altes layer: {module[j - 2]}; j: {j}')

                        seq.append(module[j - 2])
                    elif j < 2 * pos - 1:
                        # print(f'module: {module[j]}; j= {j}')
                        seq.append(module[j])
                        print(f'altes layer: {module[j - 2]}; j: {j}')

                print(f'seq: {seq}')
                newModule_list.append(nn.Sequential(*seq))
                # print(f'danach: {newModule_list[i]}')
                print(f'i: {i}')
                block = ( i + add - 2 ) % 5
                # print(f'vor Block: {block}')
                stage = (i - 2) // 5

                if ( i - 2 ) % 5 == 0 and ( i - 2 ) // 5 > 0:
                    newModule_list.append(self.module_list[i + 1])
                    m = ( ( i - 2 ) // 5 ) - 1
                    l = i - 2 * m
                    stage = ( l - 2 ) // 5

                    print(f'l: {l}; m: {m}')
                print(f'l: {l}')

                # print(f'l: {l}; i: {i - 2 + 4 * stage}')
                # if stage > 0 and block > 0:
                #     block = ( i - 2 +  4 * stage ) % 5
                if stage > 0 and block == 0:
                    blockComp = True
                    # if i % 2 == 0:
                    #     block = ( i - 2 + 4 * stage ) % 5
                    # else:
                    #     block = (i - 2 + 4 * (stage - 1)) % 5

                    # 12 -2
                print(f'Stage: {stage}; Block: {block}')
                self.archNums[stage][block] += 1
                # if (i - 2) % 5 == 0 and (i - 2) // 5 > 0:
                #     blockComp = True

            elif isinstance(self.module_list[i], nn.AdaptiveAvgPool2d):
                newModule_list.append(self.module_list[i])
            elif isinstance(self.module_list[i], nn.Linear):
                newModule_list.append(self.module_list[i])
        self.module_list = newModule_list
        print(f'Self: {self}')

        return self


def compare(layer, oddLayer):
    i1 = int(layer.split('.')[1].split('v')[1])
    i2 = int(oddLayer.split('.')[1].split('v')[1])
    if i2 + 2 == i1:
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
