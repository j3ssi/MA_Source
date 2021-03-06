
import numpy
import torch
import torch as th
import torch.nn as nn
import math
import numpy as np

from src.checkpoint_utils import makeSparse

class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock,
                 first, widthofFirstLayer=16, model=None, archNums=None, widthOfLayers=None):
        super(N2N, self).__init__()
        self.numOfStages = numOfStages
        self.oddLayers = []
        self.numOfBlocksinStage = numOfBlocksinStage
        self.layersInBlock = layersInBlock
        self.deep2 = False
        printInit = False
        if widthOfLayers is not None:
            self.widthofFirstLayer = widthOfLayers[0]
            self.widthofLayers = widthOfLayers
            if printInit:
                print(f'width: {self.widthofFirstLayer}')
        else:
            self.widthofFirstLayer = widthofFirstLayer
            self.widthofLayers = []
            s = widthofFirstLayer

            if printInit:
                print(f'numoFStages: {numOfStages}')

            for stage in range(0, numOfStages):
                self.widthofLayers.append(s)
                s *= 2
        j = 0
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
        if printInit:
            print("\nArch Num: ", self.archNums)

        self.module_list = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)

        # first Layer
        # conv1
        conv0 = nn.Conv2d(in_channels = 3, out_channels = self.widthofLayers[0], kernel_size=3, padding=1, bias=False, stride=1)
        self.module_list.append(conv0)
        if printInit:
            print(f'conv0: {conv0}; i: {j}')
        # bn1
        j += 1
        bn1 = nn.BatchNorm2d(self.widthofLayers[0])
        if printInit:
            print(f'bn1: {bn1}; i: {j}')
        j += 1
        self.module_list.append(bn1)
        self.module_list.append(self.relu)
        if printInit:
            print(f'Relu; i: {j}')

        for stage in range(0, numOfStages):
            sizeOfLayer = widthOfLayers[stage]
            # print(f'stage: {stage}; sizeof Layers: {sizeOfLayer}')
            # print("\nStage: ", stage, " ; ", sizeOfLayer)
            for block in range(0, len(self.archNums[stage])):
                layer = []
                layer2 = []
                i = 0

                while i < self.archNums[stage][block]:
                    if printInit:
                        print(f'i : {j}; block: {block}')
                    if block == 0 and stage > 0 and i == 0:
                        conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=3, padding=1,
                                             bias=False,
                                             stride=2)
                        if printInit:
                            print(f'{conv}; i={i}; if 1')
                        layer.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        if printInit:
                            print(f'{bn}; i={i}')
                        layer.append(bn)
                        layer.append(self.relu)
                        if printInit:
                            print(f'relu: {i}')
                        i = i + 1
                    elif block == 0 and stage > 0 and (i + 1) % self.archNums[stage][block] == 0:
                        conv = nn.Conv2d(int(sizeOfLayer / 2), sizeOfLayer, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=2)
                        if printInit:
                            print(f'{conv}; i: {i} if 2')
                        layer2.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        if printInit:
                            print(f'{bn}; i: {i}')
                        layer2.append(bn)
                        # layer2.append(self.relu)
                        # if printInit:
                        #    print(f'Relu; i: {i}')
                        i = i + 1
                    elif block == 0 and stage > 0 and (i + 2) % self.archNums[stage][block] == 0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                         stride=1)
                        if printInit:
                            print(f'{conv}; i: {i} if 3')
                        layer.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        if printInit:
                            print(f'{bn}; i: {i}')
                        layer.append(bn)
                        # layer.append(self.relu)
                        # if printInit:
                        #    print(f'relu; i: {i}')
                        i = i + 1
                    elif (i + 1) % self.archNums[stage][block] == 0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                         stride=1)
                        if printInit:
                            print(f'{conv}; i: {i} if 3')
                        layer.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        if printInit:
                            print(f'{bn}; i: {i}')
                        layer.append(bn)
                        # layer.append(self.relu)
                        # if printInit:
                        #    print(f'relu; i: {i}')
                        i = i + 1

                    else:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                         stride=1)
                        if printInit:
                            print(f'{conv}; i: {i} if 4')
                        layer.append(conv)
                        bn = nn.BatchNorm2d(sizeOfLayer)
                        if printInit:
                            print(f'{bn}; i: {i}')
                        layer.append(bn)
                        layer.append(self.relu)
                        if printInit:
                            print(f'relu; i: {i}')
                        i = i + 1

                    # self.paramList.append(nn.Parameter(torch.ones(1), requires_grad=True))
                    # self.paramList1.append(nn.Parameter(torch.ones(1), requires_grad=True))

                block = nn.Sequential(*layer)
                if printInit:
                    print(f'seq: {block}; i: {j}')
                j += 1
                self.module_list.append(block)
                if len(layer2) > 0:
                    block1 = nn.Sequential(*layer2)
                    self.module_list.append(block1)
                    if printInit:
                        print(f'seq1: {block1}; i: {j}')


            # print("\n self sizeofFC: ",self.sizeOfFC)


        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.module_list.append(avgpool)
        if printInit:
            print(f'avgpoll: {avgpool}')
            # 19
        fc = nn.Linear(sizeOfLayer, num_classes)
        self.module_list.append(fc)
        if printInit:
            print(f'linear: {fc}')
        for m in self.module_list:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Sequential):
                for a in range(len(m)):
                    seq = m[a]
                    if isinstance(seq, nn.Conv2d):
                        n = seq.kernel_size[0] * seq.kernel_size[1] * seq.out_channels
                        # seq.weight.data.normal_(0, math.sqrt(2. / n))
                        nn.init.normal_(seq.weight, mean=0, std=math.sqrt(2. / n))
                    elif isinstance(seq, nn.BatchNorm2d):
                        # seq.weight.data.fill_(1)
                        # seq.bias.data.zero_()
                        nn.init.ones_(seq.weight)
                        nn.init.zeros_(seq.bias)
        self.cuda()
        print(f'')
        self.StagesI, self.StagesO = self.buildResidualPath()
        # self.dense_chs, _ = makeSparse(optimizer, self, 100, reconf=False)
        # print(f'dense: {self.dense_chs}')
        # if printInit:
        print(f'Modell Erstellung')
        print(self)

    def newModuleList(self, num_classes):
        # self.sameNode, self.oddLayers = buildShareSameNodeLayers(self.module_list, self.numOfStages, self.archNums)
        # self.stageI, self.stageO = buildResidualPath(self.module_list, self.numOfStages, self.archNums)
        # print(f'sameNode: {self.sameNode}')
        # self.sameNode = model.sameNode
        # self.stageI = model.stageI
        # self.stageO = model.stageO
        # print(f'Archnums: {self.archNums}')
        module_list = self.module_list
        # del model
        self.module_list = nn.ModuleList()
        printName = False
        # print("\naltList", altList)
        for i in range(len(module_list)):
            # print("\n>i: ", i)
            print(f'module: {module_list[i]}')
            if isinstance(module_list[i], nn.Conv2d):
                print(f'Size of Weight: {module_list[i].weight.size()}')

                module = module_list[i]
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                bias = module.bias if module.bias is not None else False

                layer = nn.Conv2d(module.in_channels, module.out_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding)
                if printName:
                    print(f'layer: {layer}; i: {i}')
                layer.weight.data = module.weight.data
                self.module_list.append(layer)
            elif isinstance(module_list[i], nn.BatchNorm2d):
                print(f'Size of Weight: {module_list[i].weight.size()}')

                module = module_list[i]
                layer = nn.BatchNorm2d(module.num_features)
                layer.weight.data = module.weight.data
                layer.bias.data = module.bias.data
                if printName:
                    print(f'layer: {layer}; i: {i}')
                self.module_list.append(layer)
            elif isinstance(module_list[i], nn.Sequential):
                module = module_list[i]
                layer = []
                for j in range(len(module)):

                    if isinstance(module[j], nn.Conv2d):
                        print(f'Size of Weight: {module[j].weight.size()}')

                        module1 = module[j]
                        kernel_size = module1.kernel_size
                        stride = module1.stride
                        padding = module1.padding

                        layer1 = nn.Conv2d(module1.in_channels, module1.out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding)
                        if printName:
                            print("\n>new Layer: ", layer1)
                            # print("\nWeight Shape: ", module1.weight.shape)
                        layer1.weight.data = module1.weight.data
                        layer.append(layer1)
                    elif isinstance(module[j], nn.BatchNorm2d):
                        print(f'Size of Weight: {module[j].weight.size()}')

                        module1 = module[j]
                        layer1 = nn.BatchNorm2d(module1.num_features)
                        layer1.weight.data = module1.weight.data
                        layer1.bias.data = module1.bias.data
                        if printName:
                            print("\n>new Layer: ", layer)
                        layer.append(layer1)
                    elif isinstance(module[j], nn.ReLU):
                        layer.append(self.relu)
                self.module_list.append(nn.Sequential(*layer))
                if printName:
                    print(f'>new Layer: {layer}')
                if i == 8:
                    print(f'Sequential: {layer}')
            elif isinstance(module_list[i], nn.AdaptiveAvgPool2d):
                self.module_list.append(nn.AdaptiveAvgPool2d((1, 1)))
            elif isinstance(module_list[i], nn.Linear):
                module = module_list[i]
                print(f'Size of Weight: {module.weight.size( )}')

                fc = nn.Linear(in_features=module.in_features, out_features=num_classes)
                if printName:
                    print("\nLinear: ", fc)
                fc.weight.data = module.weight.data
                fc.bias.data = module.bias.data
                self.module_list.append(fc)
            elif isinstance(module_list[i], nn.ReLU):
                self.module_list.append(self.relu)
        if printName:
            print(f' Modell: {self}')

    def forward(self, x):
        printNet = False
        sizeofX = []
        blockNum = 0
        if printNet:
            print(f'ArchNums: {self.archNums}')
        # First layer
        if printNet:
            print("\nX Shape: ", x.shape)
        # conv1
        sizeofX.append(x)
        _x = self.module_list[0](x)
        sizeofX.append(_x)
        if printNet:
            print("\nI: 0 ; ", self.module_list[0])
            print("\n _X Shape: ", _x.shape)
        # bn1
        _x = self.module_list[1](_x)
        sizeofX.append(_x)
        if printNet:
            print("\nI: 1 ; ", self.module_list[1])
            print("\n _X Shape: ", _x.shape)
        # relu
        _x = self.module_list[2](_x)
        sizeofX.append(_x)
        j = 3
        if printNet:
            print("\nI: 2 ; ", self.module_list[2])
            print("\n _X Shape: ", _x.shape)
        # try:
        for stage in range(0, self.numOfStages):
            # if printNet:
            archNum = self.archNums[stage]
            if printNet:
                print(f'Stage: {stage}; archNum: {archNum}')
                print(f'Shape: {_x.shape}')
            for block in range(0, len(archNum)):
                # try:
                if printNet:
                    print(f'Block: {block}; j: {j}')
                seq = self.module_list[j]
                if printNet:
                    print(f'seq: {seq}')
                if block == 0 and stage > 0:
                    if printNet:
                        print(f'Drin!! seq: {seq}; j: {j}')
                    y = _x
                    z = _x
                    if printNet:
                        print(f'_x Shape: {_x.shape}; ')

                    for a in range(len(seq)):
                        y = seq[a](y)
                        sizeofX.append(y)
                        if printNet:
                            print(f'seq[a]: {seq[a]}; a: {a}')
                            print(f'y shape: {y.shape}')
                    x = y
                    block += 1
                    # x = seq(_x)
                    j += 1

                    seq = self.module_list[j]
                    if printNet:
                        print(f'Drin2!! seq: {seq}; j: {j}')

                    for a in range(len(seq)):
                        z = seq[a](z)
                        sizeofX.append(z)
                        if printNet:
                            print(f'seq[a]: {seq[a]}; a: {a}')
                            print(f'z shape: {z.shape}')
                    _x = z

                    # _x = seq(_x)
                    j += 1
                else:
                    if printNet:
                        print(f'_X Shape: {_x.shape}')
                    z = _x
                    for a in range(len(seq)):
                        z = seq[a](z)
                        sizeofX.append(z)
                        if printNet:
                            print(f'seq[a]: {seq[a]}; a: {a}')
                            print(f'z shape: {z.shape}')

                    x = z
                    j += 1

                try:
                    if self.deep2:
                        try:
                            _x = _x * self.paramList[block]
                            x = x * self.paramList1[block]
                        except:
                            if not self.deep2:
                                print(f'Fehler!')
                    _x = _x + x
                    sizeofX.append(_x)
                except RuntimeError:
                    print(f'Except')

                if printNet:
                    print(f'X Shape: {_x.shape}')
                _x = self.relu(_x)
                sizeofX.append(_x)
                # except RuntimeError:
                #     print(f'Except')
                #     print("\nJ: ", j, " ; ", self.module_list[j])
                #     print(f'seq[a]: {seq[a]}')
                #     print("\nX Shape: ", x.shape)

        if printNet:
            print("\nX Shape: ", x.shape)

        if isinstance(self.module_list[j], nn.AdaptiveAvgPool2d):
            try:
                x = self.module_list[j](_x)
                sizeofX.append(x)
                if printNet:
                    print("\nJ: ", j, " ; ", self.module_list[j])
                    print("\n\n X Shape 1: ", x.shape)
                x = x.view(x.shape[0], x.shape[1])
                sizeofX.append(x)
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
            try:
                x = self.module_list[j](x)
                sizeofX.append(x)
            except RuntimeError:
                print(f'x: {x.shape}')

            if printNet:
                print("\nJ: ", j, " ; ", self.module_list[j])
                print(f"\nsize of X: {sizeofX}")
        else:
            print("\n \n Oops!!!: ")
            print("Linear")
        return x

    def kombiPrune(self, optimizer, threshold):

        dense_chs, _ = makeSparse(optimizer, self, threshold)
        for i,j in dense_chs:
            dense = self.dense_chs


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


    def wider(self, delta_width, weight_norm=True, random_init=True,
              addNoise=True):  # teacher_w1, teacher_b1, teacher_w2, new_width, verification):
        indexL = 0
        index1 = 0
        i1 = 0
        i11 = None
        iBn1 = 0
        iBn11 = None
        i2 = 0
        i21 = None
        changeOfWidth = False
        seqIndex = 0
        printDeep = True
        finished = True
        while indexL < len(self.module_list):
            finished = True
            i = indexL
            if printDeep:
                print(f'IndexL: {indexL}')
            module = None
            moduleBn = None
            module1 = None
            if isinstance(self.module_list[indexL], nn.Conv2d):
                module = self.module_list[indexL]
                i1 = indexL
                indexL += 1
                if printDeep:
                    print(f'Module= {module}; indexL: {indexL}')
                indexConv = indexL
                while module1 is None:
                    if printDeep:
                        print(f'indexConv: {indexConv}')
                        print(f'modulelist[indexConv]: {self.module_list[indexConv]}')
                    if isinstance(self.module_list[indexConv], nn.BatchNorm2d):
                        moduleBn = self.module_list[indexConv]
                        iBn1 = indexConv
                        if printDeep:
                            print(f' moduleBn: {moduleBn}')
                    elif isinstance(self.module_list[indexConv], nn.Conv2d):
                        module1 = self.module_list[indexConv]
                        i2 = indexConv
                        if printDeep:
                            print(f'module1: {module1}; indexConv: {indexConv}; index: {index}')
                        break
                    elif isinstance(self.module_list[indexConv], nn.Sequential):
                        moduleX = self.module_list[indexConv]
                        module1 = moduleX[0]
                        i2 = indexConv
                        i21 = 0
                        indexL = indexConv
                        if printDeep:
                            print(f'module1: {module1}; indexConv: {indexConv}; index: {indexL}')

                        break
                    assert indexConv< len(self.module_list), "Falscher Index in wider"
                    indexConv += 1
            elif isinstance(self.module_list[indexL], nn.Sequential):
                moduleX = self.module_list[indexL]
                if i21 is not None:
                    i = i21
                else:
                    i = 0
                while i < len(moduleX):
                    if isinstance(moduleX[i], nn.Conv2d) and module is None:
                        module = moduleX[i]
                        i1 = indexL
                        i11 = i
                        if printDeep:
                            print(f'Module= {module}; i: {i} indexL: {indexL}')
                    elif isinstance(moduleX[i], nn.BatchNorm2d):
                        moduleBn = moduleX[i]
                        iBn1 = indexL
                        iBn11 = i
                        if printDeep:
                            print(f' moduleBn: {moduleBn}; i: {i}; indexL: {indexL}')
                    elif isinstance(moduleX[i],nn.Conv2d) and module is not None:
                        module1 = moduleX[i]
                        if printDeep:
                            print(f'module1: {module1}; i: {i}; indexL: {indexL}')
                        i2 = indexL
                        i21 = i
                        break
                    i += 1
                if module1 is None:
                    indexL += 1
                if module1 is None and isinstance(self.module_list[indexL], nn.Sequential):
                    moduleX =self.module_list[indexL]
                    i = 0
                    while i < len(moduleX):
                        if isinstance(moduleX[i], nn.Conv2d):
                            module1 = moduleX[i]
                            i2 = indexL
                            i21 = i
                            if printDeep:
                                print(f'Module= {module}; i: {i} indexL: {indexL}')
                            break
                        elif isinstance(moduleX[i], nn.BatchNorm2d):
                            moduleBn = moduleX[i]
                            iBn1 = indexL
                            iBn11 = i
                            if printDeep:
                                print(f' moduleBn: {moduleBn}; i: {i}; indexL: {indexL}')
                        i += 1
                elif isinstance(self.module_list[indexL],nn.AdaptiveAvgPool2d):
                    indexL += 1
                    module1 = self.module_list[indexL]
                    i2 = indexL
                    i21 = None
            elif isinstance(self.module_list[indexL],nn.Linear):
                module1 = self.module_list[indexL]
            else:
                indexL += 1
                finished = False

            # if not isinstance(module1,nn.Linear) and module1.in_channels != 16:
            #     continue
            # elif isinstance(module1, nn.Linear):
            #     break

            if finished and not isinstance(module1,nn.Linear):

                if module.out_channels != module1.in_channels:
                    # print(f'X!: Module: {i1}; {i11}; moduleBn: {iBn1}; {iBn11}; module1: {i2}; {i21}')
                    changeOfWidth = True
                    # module1 =None


            if not random_init and not changeOfWidth and finished and isinstance(module1, nn.Linear):

                # print(f'1: Module: {i1}; {i11}; moduleBn: {iBn1}; {iBn11}; module1: {i2}; {i21}')

                # ziehe zufällige Zahlen für die Mapping Funktion
                mapping = np.random.randint(module.out_channels, size=(int(delta_width * module.out_channels - module.out_channels)))
                # print(f'len of mapping: {len(mapping)}')

                # Ermittele wie häufig eine Zahl im Rand-Array vorhanden ist für Normalisierung
                replication_factor = np.bincount(mapping)
                # Anlage der neuen Gewichte
                new_w1 = module.weight.data.clone().cpu().detach().numpy()
                old_w1 = module.weight.data.clone().cpu().detach().numpy()

                new_w2 = module1.weight.data.clone().cpu().detach().numpy()
                old_w2 = module1.weight.data.clone().cpu().detach().numpy()

                # if module.bias is not None:
                #    new_b1 = module.weight.clone()
                #    old_b1 = module.weight.clone()

                # Fülle die neuen breiteren Gewichte mit dem richtigen Inhalt aus altem

                for i in range(len(mapping)):
                    index = mapping[i]
                    new_weight = old_w1[index, :, :, :]
                    new_weight = new_weight[np.newaxis, :, :, :]
                    new_w1 = np.concatenate((new_w1, new_weight), axis=0)
                    # if module.bias is not None:
                    #    new_b1 = np.append(new_b1, old_b1[index])

                # Fülle das Module1 mit den Gewichten un normalisiere
                for i in range(len(mapping)):
                    index = mapping[i]
                    factor = replication_factor[index] + 1
                    assert factor > 1, "Fehler in Net2Wider"
                    new_weight = old_w2[:, index] * (1. / factor)
                    new_weight_re = new_weight[:, np.newaxis]
                    new_w2 = np.concatenate((new_w2, new_weight_re), axis=1)
                    new_w2[:, index] = new_weight

                print(f'shape new w1: {new_w1.shape}')
                print(f'shape new w2: {new_w2.shape}; old w2: {old_w2.shape}')
                module.weight.data = nn.Parameter(torch.from_numpy(new_w1))
                module.out_channels = int( module.out_channels * delta_width)

                module1.weight.data = nn.Parameter( torch.from_numpy( new_w2 ) )
                module1.in_features = int( module1.in_features * delta_width )
                # print(f'module: {module}')
                # print(f'module1: {module1}')
#                if module.bias:
#                    module.bias.data = nn.Parameter(torch.from_numpy(new_b1))

                if isinstance(moduleBn, nn.BatchNorm2d):
                    print(f'Batchnorm1')
                    old_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    # print(f'len old w: {old_bn_w.size}')

                    old_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    old_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    old_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    new_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    new_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    new_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    new_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    # print(f'old weight: {old_bn_w}')
                    for i in range(0, len(mapping)):
                        index = mapping[i]
                        k = i
                        new_bn_w = np.append(new_bn_w, old_bn_w[index])
                        new_bn_b = np.append(new_bn_b, old_bn_b[index])
                        new_bn_mean = np.append(new_bn_mean, new_bn_mean[index])
                        new_bn_var = np.append(new_bn_var, new_bn_var[index])
                        # print(f'i: {i}')
                    # print(f'new bn: {new_bn_b}; K : {k}; len of bn: {new_bn_b.size}')
                    moduleBn.num_features = int( moduleBn.num_features * delta_width )
                    moduleBn.weight.data = nn.Parameter(torch.from_numpy(new_bn_w))
                    moduleBn.bias.data = nn.Parameter(torch.from_numpy(new_bn_b))
                    moduleBn.running_mean = torch.from_numpy(new_bn_mean)
                    moduleBn.running_var = torch.from_numpy(new_bn_var)
            elif not random_init and not changeOfWidth and finished:

                # print(f'1: Module: {i1}; {i11}; moduleBn: {iBn1}; {iBn11}; module1: {i2}; {i21}')

                # ziehe zufällige Zahlen für die Mapping Funktion
                mapping = np.random.randint(module.out_channels,
                                            size=(int( delta_width * module.out_channels - module.out_channels ) ) )
                # print(f'len of mapping: {len(mapping)}')

                # Ermittele wie häufig eine Zahl im Rand-Array vorhanden ist für Normalisierung
                replication_factor = np.bincount(mapping)
                # Anlage der neuen Gewichte
                new_w1 = module.weight.data.clone().cpu().detach().numpy()
                new_w2 = module1.weight.data.clone().cpu().detach().numpy()
                old_w1 = module.weight.data.clone().cpu().detach().numpy()
                old_w2 = module1.weight.data.clone().cpu().detach().numpy()

                if module.bias is not None:
                    new_b1 = module.weight.clone()
                    old_b1 = module.weight.clone()
                if module1.bias is not None:
                    new_b2 = module1.weight.clone()
                    old_b2 = module1.weight.clone()

                # Fülle die neuen breiteren Gewichte mit dem richtigen Inhalt aus altem
                for i in range(len(mapping)):
                    index = mapping[i]
                    new_weight = old_w1[index, :, :, :]
                    new_weight = new_weight[np.newaxis, :, :, :]
                    new_w1 = np.concatenate((new_w1, new_weight), axis=0)
                    #if module.bias is not None:
                    #    new_b1 = np.append(new_b1, old_b1[index])

                # Fülle das Module1 mit den Gewichten un normalisiere
                for i in range(len(mapping)):
                    index = mapping[i]
                    factor = replication_factor[index] + 1
                    assert factor > 1, "Fehler in Net2Wider"
                    new_weight = old_w2[:, index, :, :] * (1. / factor)
                    new_weight_re = new_weight[:, np.newaxis, :, :]
                    new_w2 = np.concatenate((new_w2, new_weight_re), axis=1)
                    new_w2[:, index, :, :] = new_weight

                print(f'shape new w1: {new_w1.shape}')
                print(f'shape new w2: {new_w2.shape}; old w2: {old_w2.shape}')
                module.weight.data = nn.Parameter(torch.from_numpy(new_w1))
                module.out_channels = int( module.out_channels * delta_width )

                module1.weight.data = nn.Parameter(torch.from_numpy(new_w2))
                module1.in_channels = int( module1.in_channels * delta_width )
                # print(f'module: {module}')
                # print(f'module1: {module1}')
                # if module.bias:
                #    module.bias.data = nn.Parameter(torch.from_numpy(new_b1))

                if isinstance(moduleBn, nn.BatchNorm2d):
                    print(f'Batchnorm1')
                    old_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    # print(f'len old w: {old_bn_w.size}')

                    old_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    old_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    old_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    new_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    new_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    new_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    new_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    # print(f'old weight: {old_bn_w}')
                    for i in range(0, len(mapping)):
                        index = mapping[i]
                        k = i
                        new_bn_w = np.append(new_bn_w, old_bn_w[index])
                        new_bn_b = np.append(new_bn_b, old_bn_b[index])
                        new_bn_mean = np.append(new_bn_mean, new_bn_mean[index])
                        new_bn_var = np.append(new_bn_var, new_bn_var[index])
                        # print(f'i: {i}')
                    # print(f'new bn: {new_bn_b}; K : {k}; len of bn: {new_bn_b.size}')
                    moduleBn.num_features = int(moduleBn.num_features * delta_width)
                    moduleBn.weight.data = nn.Parameter(torch.from_numpy(new_bn_w))
                    moduleBn.bias.data = nn.Parameter(torch.from_numpy(new_bn_b))
                    moduleBn.running_mean = torch.from_numpy(new_bn_mean)
                    moduleBn.running_var = torch.from_numpy(new_bn_var)
            elif not random_init and finished:
                # # print(f'2: Module: {i1}; {i11}; moduleBn: {iBn1}; {iBn11}; module1: {i2}; {i21}')
                #
                # # ziehe zufällige Zahlen für die Mapping Funktion
                mapping = np.random.randint(module.weight.size(0),
                                            size=(int( delta_width * module.weight.size(0) - module.weight.size(0))))
                # print(f'len of mapping: {len(mapping)}')

                # Ermittele wie häufig eine Zahl im Rand-Array vorhanden ist für Normalisierung
                replication_factor = np.bincount(mapping)
                # Anlage der neuen Gewichte
                new_w1 = module.weight.data.clone().cpu().detach().numpy()
                old_w1 = module.weight.data.clone().cpu().detach().numpy()

                if module.bias is not None:
                    new_b1 = module.weight.clone()
                    old_b1 = module.weight.clone()

                # Fülle die neuen breiteren Gewichte mit dem richtigen Inhalt aus altem
                for i in range(len(mapping)):
                    index = mapping[i]
                    new_weight = old_w1[index, :, :, :]
                    new_weight = new_weight[np.newaxis, :, :, :]
                    new_w1 = np.concatenate((new_w1, new_weight), axis=0)
                    # if module.bias is not None:
                    #    new_b1 = np.append(new_b1, old_b1[index])
                print(f'shape new w1: {new_w1.shape}')
                module.weight.data = nn.Parameter(torch.from_numpy(new_w1))
                module.out_channels = int( module.out_channels * delta_width )


                print(f'module: {module}')
                # if module.bias:
                #    module.bias.data = nn.Parameter(torch.from_numpy(new_b1))

                if isinstance(moduleBn, nn.BatchNorm2d):
                    print(f'Batchnorm2')
                    old_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    # print(f'len old w: {old_bn_w.size}')

                    old_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    old_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    old_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    new_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    new_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    new_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    new_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    # print(f'old weight: {old_bn_w}')
                    for i in range(0, len(mapping)):
                        index = mapping[i]
                        k = i
                        new_bn_w = np.append(new_bn_w, old_bn_w[index])
                        new_bn_b = np.append(new_bn_b, old_bn_b[index])
                        new_bn_mean = np.append(new_bn_mean, new_bn_mean[index])
                        new_bn_var = np.append(new_bn_var, new_bn_var[index])
                        # print(f'i: {i}')
                    # print(f'new bn: {new_bn_b}; K : {k}; len of bn: {new_bn_b.size}')
                    moduleBn.num_features = int( moduleBn.num_features * delta_width)
                    moduleBn.weight.data = nn.Parameter(torch.from_numpy(new_bn_w))
                    moduleBn.bias.data = nn.Parameter(torch.from_numpy(new_bn_b))
                    moduleBn.running_mean = torch.from_numpy(new_bn_mean)
                    moduleBn.running_var = torch.from_numpy(new_bn_var)

                print(f'1: Module: {i1}; {i11}; moduleBn: {iBn1}; {iBn11}; module1: {i2}; {i21}')

                # ziehe zufällige Zahlen für die Mapping Funktion
                mapping = np.random.randint(module1.in_channels,
                                            size=(int(delta_width * module1.in_channels - module1.in_channels)))
                # print(f'len of mapping: {len(mapping)}')

                # Ermittele wie häufig eine Zahl im Rand-Array vorhanden ist für Normalisierung
                replication_factor = np.bincount(mapping)
                # Anlage der neuen Gewichte
                new_w2 = module1.weight.data.clone().cpu().detach().numpy()
                old_w2 = module1.weight.data.clone().cpu().detach().numpy()

                # Fülle das Module1 mit den Gewichten un normalisiere
                for i in range(len(mapping)):
                    index = mapping[i]
                    factor = replication_factor[index] + 1
                    assert factor > 1, "Fehler in Net2Wider"
                    new_weight = old_w2[:, index, :, :] * (1. / factor)
                    new_weight_re = new_weight[:, np.newaxis, :, :]
                    new_w2 = np.concatenate((new_w2, new_weight_re), axis=1)
                    new_w2[:, index, :, :] = new_weight
                print(f'shape new w2: {new_w2.shape}')

                module1.weight.data = nn.Parameter(torch.from_numpy(new_w2))
                module1.in_channels = int( module1.in_channels * delta_width )
                # print(f'module: {module}')
                # print(f'module1: {module1}')
                changeOfWidth = False
            elif random_init:
                i0 = int( module.out_channels * (delta_width - 1) )
                iX = module.out_channels
                i1 = module.in_channels
                i2 = module.kernel_size[0]
                i3 = module.kernel_size[1]
                old_w1 = module.weight.data.clone().cpu().detach().numpy()
                print(f'dtype tensor: {module.weight.data.dtype}; dtype numpy: {old_w1.dtype}')
                # print(f'size of weight before: {module.weight.size()}')
                n = i2 * i3 * module.out_channels
                new_w1 = n * np.random.randn(i0, i1, i2, i3)
                print(f'dtype new tensor: {new_w1.dtype}')
                print(f'oldw shape: {old_w1.shape}; new shape: {new_w1.shape}')
                new_w1 = np.concatenate((old_w1, new_w1), axis = 0 )
                new_w1 = new_w1.astype('float32')
                print(f'new shape: {new_w1.shape}')
                # for k in range(0, module.out_channels):
                #     new_w1[k, :, :, :] = old_w1[k, :, :, :]
                module.out_channels = int( module.out_channels * delta_width)
                module.weight.data = torch.from_numpy(new_w1)
                print(f'module after: {module}')
                print(f'size of weight after: {module.weight.size()}')


                if isinstance(moduleBn, nn.BatchNorm2d):
                    mapping = np.random.randint(iX,
                                                size=(int((delta_width - 1) * iX)))
                    print(f'Batchnorm1')
                    old_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    # print(f'len old w: {old_bn_w.size}')

                    old_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    new_bn_w = moduleBn.weight.data.clone().cpu().detach().numpy()
                    new_bn_b = moduleBn.bias.data.clone().cpu().detach().numpy()
                    new_bn_mean = moduleBn.running_mean.clone().cpu().detach().numpy()
                    new_bn_var = moduleBn.running_var.clone().cpu().detach().numpy()
                    # print(f'old weight: {old_bn_w}')
                    for i in range(0, len(mapping)):
                        index = mapping[i]
                        new_bn_w = np.append(new_bn_w, old_bn_w[index])
                        new_bn_b = np.append(new_bn_b, old_bn_b[index])
                        new_bn_mean = np.append(new_bn_mean, new_bn_mean[index])
                        new_bn_var = np.append(new_bn_var, new_bn_var[index])
                        # print(f'i: {i}')
                    # print(f'new bn: {new_bn_b}; K : {k}; len of bn: {new_bn_b.size}')
                    moduleBn.num_features = int(moduleBn.num_features * delta_width)
                    moduleBn.weight.data = nn.Parameter(torch.from_numpy(new_bn_w))
                    moduleBn.bias.data = nn.Parameter(torch.from_numpy(new_bn_b))
                    moduleBn.running_mean = torch.from_numpy(new_bn_mean)
                    moduleBn.running_var = torch.from_numpy(new_bn_var)

                if isinstance(module1, nn.Conv2d):
                    i0 = module1.out_channels
                    i1 = int( module1.in_channels * (delta_width - 1))
                    i2 = module1.kernel_size[0]
                    i3 = module1.kernel_size[1]
                    old_w2 = module1.weight.data.clone().cpu().detach().numpy()
                    n = i2 * i3 * module1.in_channels
                    new_w2 = n * np.random.randn(i0, i1, i2, i3)
                    new_w2 = new_w2.astype('float32')

                    print(f'oldw2 shape: {old_w2.shape}; new shape w2: {new_w2.shape}')
                    new_w2 = np.concatenate((old_w2, new_w2), axis = 1)
                    print(f'new shape: {new_w2.shape}')

                    # for k in range(0, module1.in_channels):
                    #     new_w2[:, k, :, :] = old_w2[:, k, :, :]
                    module1.in_channels = int(module1.in_channels * delta_width)
                    module1.weight.data = torch.from_numpy(new_w2)
                    print(f'module1 after: {module1}')
                elif isinstance(module1, nn.Linear):
                    i0 = module1.out_features
                    i1 = int(module1.in_features * (delta_width -1))
                    old_w2 = module1.weight.data.clone().cpu().detach().numpy()
                    n = i1

                    new_w2 = n * np.random.randn(i0, i1)
                    new_w2 = new_w2.astype('float32')

                    print(f'oldw2 shape: {old_w2.shape}; new shape w2: {new_w2.shape}')
                    new_w2 = np.concatenate((old_w2,new_w2), axis = 1)
                    print(f'new shape: {new_w2.shape}')

                    # for k in range(0, module1.in_features):
                    #     new_w2[:, k] = old_w2[:, k]
                    module1.in_features = int(module1.in_features * delta_width)
                    module1.weight.data = torch.from_numpy(new_w2)
                    print(f'module after: {module1}')
                    print(f'size of weight after: {module1.weight.size()}')

            if isinstance(module1, nn.Linear):
                break
        print(f'self: {self}')

    def deeper(self, pos=1):
        # make each block with plus two layers (conv +batch) deeper
        blockComp = False
        for i in range(0, len(self.module_list)):
            module = self.module_list[i]
            if i > 4 and blockComp and isinstance(module, nn.Sequential):
                print(f'skip: {i}')
                blockComp = False
                continue

            if isinstance(module, nn.Sequential):
                print(f'i: {i}')

                i0 = module[0].out_channels
                i1 = module[0].in_channels
                seq = []

                for j in range(0, len(module) + 3):
                    print(f'j: {j}; ')
                    if j == (3 * pos - 2):
                        bn = nn.BatchNorm2d(module[0].out_channels)
                        torch.nn.init.ones_(bn.weight)
                        torch.nn.init.zeros_(bn.bias)
                        seq.append(bn)
                        # print(f'neues bn: {bn}; j: {j}')
                    elif j == (3 * pos - 1):
                        seq.append(self.relu)
                    elif j == (3 * pos):
                        # continue
                        kernel_size = module[0].kernel_size
                        stride = 1
                        padding = 1
                        conv = nn.Conv2d(i0, i0, kernel_size=kernel_size, stride=stride, padding=padding)

                        deeper_w = np.zeros((i0, i0, kernel_size[0], kernel_size[1]))
                        # deeper_w = np.random.normal(loc=0, scale=0.15,size=(i0, i0, kernel_size[0], kernel_size[1]))
                        for a1 in range(0, i0):
                            for b1 in range(0, i0):
                                deeper_w[a1, b1, 1, 1]=1
                                # print(f'deeper w :{deeper_w[i][j]}')
                        deeper_w = torch.from_numpy(deeper_w)
                        conv.weight.data = deeper_w.type(torch.FloatTensor)
                        # nn.init.normal_( conv.weight, mean = 0, std = math.sqrt( 1. / ( n*n ) ) )
                        # lastConv = False
                        # k = 1
                        # while not lastConv:
                        #     if ( j - k ) >= len(module):
                        #         # print(f'k: {k}; j-k: {j-k}; len(module): {len(module)}')
                        #         k += 1
                        #     else:
                        #         m = module[ j - k ]
                        #         if isinstance(m, nn.Conv2d):
                        #             for l in range(m.out_channels):
                        #                 weight1 = m.weight.data
                        #                 norm = weight1.select(0, l).norm()
                        #                 weight1.div_(norm)
                        #                 m.weight.data = weight1
                        #             module[ j - k ] = m
                        #             lastConv = True
                        #         else:
                        #             k += 1
                        seq.append(conv)
                    elif j > (3 * pos):
                        # print(f'module: {module[j - 2]}; j= {j + 2}')
                        # print(f'altes layer: {module[j - 3]}; j: {j}')
                        # if isinstance(module[ j - 3 ], nn.Conv2d):
                        #     n = module[ j - 3 ].kernel_size[0] * module[ j - 3 ].kernel_size[1] * module[ j - 3 ].out_channels
                        #     nn.init.normal_(module[ j - 3 ].weight, mean=0, std=math.sqrt(2. / (n)))
                        # elif isinstance(module[ j - 3 ], nn.BatchNorm2d):
                        #     torch.nn.init.ones_( module[ j - 3 ].weight )
                        #     torch.nn.init.zeros_( module[ j - 3 ].bias )

                        seq.append(module[j - 3])
                    elif j < 3 * pos - 2:
                        # prin   t(f'module: {module[j]}; j= {j}')
                        # if isinstance(module[j], nn.Conv2d):
                        #     n = module[j].kernel_size[0] * module[j].kernel_size[1] * module[j].out_channels
                        #     nn.init.normal_(module[j].weight, mean=0, std=math.sqrt(2. / (n)))
                        # elif isinstance(module[j], nn.BatchNorm2d):
                        #     torch.nn.init.ones_(module[j].weight)
                        #     torch.nn.init.zeros_(module[j].bias)

                        seq.append(module[j])

                        # print(f'altes layer: {module[j]}; j: {j}')
                    print(f'seq: {seq}')
                    print(f'seq[j]: {seq[j]}')
                print(f'i: {i}; i0=: {i0}; i1=: {i1}')
                self.module_list[i] = nn.Sequential(*seq)
                if i0 != i1 and not blockComp:
                    blockComp = True
        print(self)

    def deeper2(self, pos):
        # make each stage with one block more
        print(f'deep2: {self.deep2}')
        if not self.deep2:
            self.deep2 = True
            self.paramList = nn.ParameterList()
            self.paramList1 = nn.ParameterList()
            for stage in range(0, self.numOfStages):
                for block in range(0, len(self.archNums[stage])):
                    self.paramList.append(nn.Parameter(torch.ones(1), requires_grad=True))
                    self.paramList1.append(nn.Parameter(torch.ones(1), requires_grad=True))
        print(f'len param: {len(self.paramList)}')
        printDeeper = False
        j = 2
        notfirstStage = False

        # if printDeeper:
        #    print("\n\nStage: ", stage)
        # archNum = self.archNums[stage]
        firstBlockInStage = True
        paramListTmp = nn.ParameterList()
        paramListTmp1 = nn.ParameterList()
        moduleList = nn.ModuleList()
        moduleList.append(self.module_list[0])
        moduleList.append(self.module_list[1])
        moduleList.append(self.module_list[2])

        k = 0
        j = 3
        for stage in range(0, self.numOfStages):
            if isinstance(self.module_list[j], nn.Sequential):
                module = self.module_list[j]
                module = module[0]
                i0 = module.out_channels
                i1 = module.weight.size(1)
                i2 = module.weight.size(2)
                i3 = module.weight.size(3)
                if printDeeper:
                    print(f'size: {i0}, {i1}, {i2}, {i3}; j: {j}')
                for block in range(0, len(self.archNums[stage])+ 1):
                    print(f'Block: {block}')
                    if block < pos and stage > 0 and block == 0 :
                        moduleList.append(self.module_list[j])
                        moduleList.append(self.module_list[j + 1])
                        paramListTmp.append(nn.Parameter(self.paramList[k], requires_grad=True))
                        paramListTmp1.append(nn.Parameter(self.paramList1[k], requires_grad = True))
                        k += 1
                        j += 2
                    elif block < pos:
                        moduleList.append(self.module_list[j])
                        paramListTmp.append(nn.Parameter(self.paramList[k], requires_grad=True))
                        paramListTmp1.append(nn.Parameter(self.paramList1[k], requires_grad=True))
                        k += 1
                        j += 1
                    elif block == pos:
                        if stage>0:
                            numOfBlocks = self.archNums[stage][0]-1
                        else:
                            numOfBlocks = self.archNums[stage][0]
                        self.archNums[stage].insert(k, numOfBlocks)
                        param1 = nn.Parameter(torch.ones(1))
                        param1.data.fill_(0.5)
                        paramListTmp.append(param1)
                        paramListTmp1.append(param1)
                        layer = []
                        i = 0
                        sizeOfLayer = self.widthofLayers[stage]
                        while i < numOfBlocks:
                            if printDeeper:
                                print(f'i : {j}; block: {block}')
                            if (i + 1) % self.archNums[stage][block] == 0:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                                 stride=1)
                                deeper_w = np.zeros((i0, i0, i2, i3))
                                # deeper_w = np.random.normal(loc=0, scale=0.15,size=(i0, i0, kernel_size[0], kernel_size[1]))
                                for a1 in range(0, i0):
                                    for b1 in range(0, i0):
                                        deeper_w[a1, b1, 1, 1] = 1
                                        # print(f'deeper w :{deeper_w[i][j]}')
                                deeper_w = torch.from_numpy(deeper_w)
                                conv.weight.data = deeper_w.type(torch.FloatTensor)

                                if printDeeper:
                                    print(f'{conv}; i: {i} if 3')
                                layer.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                torch.nn.init.ones_(bn.weight)
                                torch.nn.init.zeros_(bn.bias)

                                if printDeeper:
                                    print(f'{bn}; i: {i}')
                                layer.append(bn)
                                # layer.append(self.relu)
                                # if printInit:
                                #    print(f'relu; i: {i}')
                                i = i + 1

                            else:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1, bias=False,
                                                 stride=1)
                                if printDeeper:
                                    print(f'{conv}; i: {i} if 4')
                                layer.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                torch.nn.init.ones_(bn.weight)
                                torch.nn.init.zeros_(bn.bias)
                                if printDeeper:
                                    print(f'{bn}; i: {i}')
                                layer.append(bn)
                                layer.append(self.relu)
                                if printDeeper:
                                    print(f'relu; i: {i}')
                                i = i + 1

                        block = nn.Sequential(*layer)
                        if printDeeper:
                            print(f'seq: {block}; i: {j}')
                        moduleList.append(block)
                    elif block > pos:
                        paramListTmp.append(nn.Parameter(self.paramList[k], requires_grad=True))
                        paramListTmp1.append(nn.Parameter(self.paramList1[k], requires_grad=True))
                        moduleList.append(self.module_list[j])
                        k += 1
                        j += 1
        # b = 2
        # c = 0
        # for i in range(0, stage - 1):
        #     archStage = self.archNums[i - 1]
        #     print(f'archStage: {archStage}')
        #     for j in range(len(archStage)):
        #         b += 2 * archStage[j - 1]
        #         c = c + 1
        # archStage = self.archNums[stage - 1]
        # for i in range(0, pos):
        #     b += 2 * pos
        #     c = c + 1
        # print(f'B: {b}; c: {c}')
        # paramListTmp1 = nn.ParameterList()
        # paramListTmp = nn.ParameterList()
        # for i in range(len(self.paramList)):
        #
        #     if i == c:
        #         param1 = nn.Parameter(torch.ones(1))
        #         param1.data.fill_(0.5)
        #         paramListTmp.append(param1)
        #         paramListTmp1.append(param1)
        #     paramListTmp.append(self.paramList[i])
        #     paramListTmp1.append(self.paramList1[i])

        # print(f'paramlist: {paramListTmp}')
        # print(f'paramlist1: {paramListTmp1}')
        print(f'archNums: {self.archNums}')
        print(f'len paramList: {len(paramListTmp)}')

        avgpool = nn.AdaptiveAvgPool2d((1, 1))
        moduleList.append(avgpool)
        moduleList.append(self.module_list[-1])
        self.module_list = moduleList
        self.paramList = paramListTmp
        self.paramList1 = paramListTmp1
        self.paramList.cuda()
        self.paramList1.cuda()
        print(self)

    def buildResidualPath(self,):
        # # stage0O = [n(1), n(3), n(5), n(7), n(9), n(11)]
    # # stages1O = [n(13), n(14), n(16), n(18), n(20), n(22)]
    # # stages2O = [n(24), n(25), n(27), n(29), n(31), n(33)]
    # printStages = False
    # sameNode, oddLayers = buildShareSameNodeLayers(module_list, numOfStages, archNums)
    # tempStagesI = []
    # tempStagesO = [n(1)]
    # stageWidth = module_list[0].weight.size()[0]
    # oddLayersCopy = oddLayers
    # oddLayersBool = False
    # for node in sameNode:
    #     if len(oddLayers) > 0:
    #         # print(f'oddLayer: {self.oddLayers[0]}')
    #         if compare(node[-1], oddLayers[0]):
    #             oddLayer = oddLayers.pop(0)
    #             tempStagesO.append(oddLayer)
    #             tempStagesI.append(oddLayer)
    #             oddLayersBool = True
    #     tempStagesI.append(node[0])
    #     tempStagesO.append(node[-1])
    #
    # length = len(module_list)
    # fcStr = 'fc' + str(int(length / 2))
    # tempStagesI.append(n(fcStr))
    # stagesI = [[]]
    # stagesO = [[]]
    # for layer in tempStagesI:
    #     # print(layer)
    #     if 'conv' in layer:
    #         i = int(layer.split('.')[1].split('v')[1])
    #         i = 2 * i - 2
    #         if i == 0:
    #             stagesI[0].append(layer)
    #         elif module_list[i].weight.size()[1] == stageWidth:
    #             stagesI[-1].append(layer)
    #         else:
    #             stageWidth = module_list[i].weight.size()[1]
    #             stagesI.append([])
    #             stagesI[-1].append(layer)
    #
    #     elif 'fc' in layer:
    #         stagesI[-1].append(layer)
    #     # print(f'StagesI:{stagesI}')
    #
    # stageWidth = module_list[0].weight.size()[0]
    # for layer in tempStagesO:
    #     # print(layer)
    #     i = int(layer.split('.')[1].split('v')[1])
    #     i = 2 * i - 2
    #     if module_list[i].weight.size()[0] == stageWidth:
    #         stagesO[-1].append(layer)
    #     elif layer in oddLayersCopy:
    #         stagesO[1].append(layer)
    #     else:
    #         stageWidth = module_list[i].weight.size()[0]
    #         stagesO.append([])
    #         stagesO[-1].append(layer)
    #
    # # print(f'stagesI: {stagesI}')
    #
    # # print(f'stagesO: {stagesO}')
        stagesI, stagesO = {}, {}

        for width in self.widthofLayers:
            print(f'width: {width}')
            k = 0
            for module in self.module_list:
                print(f'module: {module}')
                if isinstance(module, nn.Sequential):
                    print(f'width Module: {module[0].in_channels}')

                    if module[0].in_channels == width:
                        if width in stagesI.keys():
                            stagesI[width].append((k,0))
                        else:
                            stagesI[width]= [(k,0)]
                    j = - 1

                    while j < 0:
                        if isinstance(module[j],nn.Conv2d):
                            if module[j].out_channels == width:
                                print(f'conv gefunden')
                                if width in stagesO.keys():
                                    print(f'(i,j): ({k}, {len(module) + j}')
                                    stagesO[width].append((k, len(module) + j))
                                else:
                                    stagesO[width] = [(k, len(module) + j)]#                     j = 1
                            j = 1
                        else:
                            j= j-1
                elif isinstance(module, nn.Conv2d):
                    if module.out_channels == width:
                        stagesO[width] = [(0,None)]
                elif isinstance(module, nn.Linear):
                    if module.in_features == width:
                        stagesI[width].append((k, None))

                k += 1

            #                 else:
            #                     j = j - 1
            #
            #     elif isinstance(module, nn.Conv2d):
            #         print(f'module')

        print(f'stagesI: {stagesI}')
        print(f'stagesO: {stagesO}')
        return stagesI, stagesO

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
