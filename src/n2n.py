import copy
import torch
import torch.nn as nn
import math


class N2N(nn.Module):

    def __init__(self, num_classes, numOfStages, numOfBlocksinStage, layersInBlock, first, bottleneck, model=None):
        super(N2N, self).__init__()
        self.numOfStages = numOfStages
        self.numOfBlocksinStage = numOfBlocksinStage
        self.bottleneck = bottleneck
        self.layersInBlock = layersInBlock
        if first:
            self.archNums = [[]]
            for s in range(0, self.numOfStages):
                print("\nS: ", s, " ; ", self.numOfStages)
                for b in range(0, self.numOfBlocksinStage[s]):
                    self.archNums[s].append(self.layersInBlock)
                if s != (self.numOfStages - 1):
                    self.archNums.append([])
            print("\nArch Num: ", self.archNums)

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
                    print("\nStage: ", stage, " ; ", sizeOfLayer)
                    #CONV 1
                    if stage == 0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)
                    else:
                        conv = nn.Conv2d(sizeOfLayer * 2, sizeOfLayer, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)
                    print(f'list length: {len(self.module_list)}')
                    print(f'conv: {conv}')
                    self.module_list.append(conv)
                    #bn1
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

                    conv = nn.Conv2d(sizeOfLayer, sizeOfLayer*4, kernel_size=1, padding=0,
                                     bias=False,
                                     stride=1)
                    self.module_list.append(conv)
                    bn = nn.BatchNorm2d(sizeOfLayer*4)
                    self.module_list.append(bn)
                    if stage==0:
                        conv = nn.Conv2d(sizeOfLayer, sizeOfLayer*4, kernel_size=3, padding=1,
                                     bias=False,
                                     stride=1)
                    else:
                        conv = nn.Conv2d(sizeOfLayer*2, sizeOfLayer*4, kernel_size=1, padding=0,
                                         bias=False,
                                         stride=1)

                    self.module_list.append(conv)
                    bn = nn.BatchNorm2d(sizeOfLayer*4)
                    self.module_list.append(bn)

                    print(f'archNums: {len(self.archNums[stage-1])}')
                    for i in range(0, len(self.archNums[stage-1])-1):
                        j=0
                        while j< self.archNums[stage-1][i+1]:
                            print(f'self.archNums[stage-1][i+1]:{self.archNums[stage-1][i+1]}')
                            if (j == 0):
                                conv = nn.Conv2d(sizeOfLayer * 4, sizeOfLayer, kernel_size=1, padding=0,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                j=j+1
                            elif (j + 1) % self.archNums[stage - 1][i + 1] != 0:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer, kernel_size=3, padding=1,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(sizeOfLayer)
                                self.module_list.append(bn)
                                j=j+1
                            else:
                                conv = nn.Conv2d(sizeOfLayer, sizeOfLayer * 4, kernel_size=1, padding=0,
                                                 bias=False,
                                                 stride=1)
                                self.module_list.append(conv)
                                bn = nn.BatchNorm2d(4 * sizeOfLayer)
                                self.module_list.append(bn)
                                j=j+1

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
            print(self)
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
            archNum = self.archNums[stage - 1]
            firstLayerInStage = True
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
                        x=self.relu(x)

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
                        x=self.relu(x)

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
                                print("\nJ: ", j, " ; ", self.module_list[j])
                                print("\nX Shape: ", x.shape)
                            j = j + 1

                            # bn
                            x = self.module_list[j](x)
                            if printNet:
                                print("\nJ: ", j, " ; ", self.module_list[j])
                            j = j + 1
                            i=i+1
                            x = self.relu(x)

                            if ((i + 1) % self.layersInBlock) == 0:

                                if printNet:
                                    print("\nShortcutLayer J: ", j, " ; ", self.module_list[j])
                                j = j + 1

                                _x = _x + x
                                _x = self.relu(_x)


                            # elif ((i + 1) % self.layersInBlock) == 0:
                            #
                            #     # conv
                            #     x = self.module_list[j](x)
                            #     if printNet:
                            #         print("\nJ: ", j, " ; ", self.module_list[j])
                            #         print("\nX Shape: ", x.shape)
                            #     j = j + 1
                            #
                            #     # bn
                            #     x = self.module_list[j](x)
                            #     if printNet:
                            #         print("\nJ: ", j, " ; ", self.module_list[j])
                            #     j = j + 1
                            #     i = i + 1
                            #     _x = self.relu(x)
                            #
                            # else:
                            #     # conv
                            #     x = self.module_list[j](x)
                            #     if printNet:
                            #         print("\nJ: ", j, " ; ", self.module_list[j])
                            #         print("\nX Shape: ", x.shape)
                            #     j = j + 1
                            #     # bn
                            #     x = self.module_list[j](x)
                            #     if printNet:
                            #         print("\nJ: ", j, " ; ", self.module_list[j])
                            #         print("\nX Shape: ", x.shape)
                            #     j = j + 1
                            #     x = self.relu(x)
                            #     i = i + 1


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
        stagesI = []
        stagesO = []
        i = 1
        printStages = False
        stagesI.append([])
        stagesO.append([])
        stagesO[0].append(n(1))
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
                    i = i + layerInThisBlock - 1
                    stagesO[-1].append(n(i))
                elif block == 0:
                    layerInThisBlock = archNum[block]
                    i = i + layerInThisBlock - 1
                    stagesO[-1].append(n(i))

        # print("\nstagesO:  1")
        printStages = False
        fcStr = 'fc' + str(i + 1)
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

        print("\nSame Node: ", sameNode)
        return sameNode

    def wider(model):
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

        """
        Convert m1 layer to its wider version by adapthing next weight layer and
        possible batch norm layer in btw.
        Args:
            m1 - module to be wider
            m2 - follwing module to be adapted to m1
            new_width - new width for m1.
            bn (optional) - batch norm layer, if there is btw m1 and m2
            out_size (list, optional) - necessary for m1 == conv3d and m2 == linear. It
            is 3rd dim size of the output feature map of m1. Used to compute
            the matching Linear layer size
        """
        #     sameNodes=model.getShareSameNodeLayers()
        #     for listNodes in sameNodes:
        #         if :
        #             for elem in listNodes:
        #
        #
        #
        # w1 = m1.weight.data
        # w2 = m2.weight.data
        # b1 = m1.bias.data
        #
        # if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        #     # Convert Linear layers to Conv if linear layer follows target layer
        #     if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
        #         assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
        #         if w1.dim() == 4:
        #             factor = int(np.sqrt(w2.size(1) // w1.size(0)))
        #             w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
        #         elif w1.dim() == 5:
        #             assert out_size is not None,\
        #                    "For conv3d -> linear out_size is necessary"
        #             factor = out_size[0] * out_size[1] * out_size[2]
        #             w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
        #                          out_size[1], out_size[2])
        #     else:
        #         assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        #     assert new_width > w1.size(0), "New size should be larger"
        #
        #     old_width = w1.size(0)
        #     nw1 = m1.weight.data.clone()
        #     nw2 = w2.clone()
        #
        #     if nw1.dim() == 4:
        #         nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3))
        #         nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3))
        #     elif nw1.dim() == 5:
        #         nw1.resize_(new_width, nw1.size(1), nw1.size(2), nw1.size(3), nw1.size(4))
        #         nw2.resize_(nw2.size(0), new_width, nw2.size(2), nw2.size(3), nw2.size(4))
        #     else:
        #         nw1.resize_(new_width, nw1.size(1))
        #         nw2.resize_(nw2.size(0), new_width)
        #
        #     if b1 is not None:
        #         nb1 = m1.bias.data.clone()
        #         nb1.resize_(new_width)
        #
        #     if bnorm is not None:
        #         nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
        #         nrunning_var = bnorm.running_var.clone().resize_(new_width)
        #         if bnorm.affine:
        #             nweight = bnorm.weight.data.clone().resize_(new_width)
        #             nbias = bnorm.bias.data.clone().resize_(new_width)
        #
        #     w2 = w2.transpose(0, 1)
        #     nw2 = nw2.transpose(0, 1)
        #
        #     nw1.narrow(0, 0, old_width).copy_(w1)
        #     nw2.narrow(0, 0, old_width).copy_(w2)
        #     nb1.narrow(0, 0, old_width).copy_(b1)
        #
        #     if bnorm is not None:
        #         nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
        #         nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
        #         if bnorm.affine:
        #             nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
        #             nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)
        #
        #     # TEST:normalize weights
        #     if weight_norm:
        #         for i in range(old_width):
        #             norm = w1.select(0, i).norm()
        #             w1.select(0, i).div_(norm)
        #
        #     # select weights randomly
        #     tracking = dict()
        #     for i in range(old_width, new_width):
        #         idx = np.random.randint(0, old_width)
        #         try:
        #             tracking[idx].append(i)
        #         except:
        #             tracking[idx] = [idx]
        #             tracking[idx].append(i)
        #
        #         # TEST:random init for new units
        #         if random_init:
        #             n = m1.kernel_size[0] * m1.kernel_size[1] * m1.out_channels
        #             if m2.weight.dim() == 4:
        #                 n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.out_channels
        #             elif m2.weight.dim() == 5:
        #                 n2 = m2.kernel_size[0] * m2.kernel_size[1] * m2.kernel_size[2] * m2.out_channels
        #             elif m2.weight.dim() == 2:
        #                 n2 = m2.out_features * m2.in_features
        #             nw1.select(0, i).normal_(0, np.sqrt(2./n))
        #             nw2.select(0, i).normal_(0, np.sqrt(2./n2))
        #         else:
        #             nw1.select(0, i).copy_(w1.select(0, idx).clone())
        #             nw2.select(0, i).copy_(w2.select(0, idx).clone())
        #         nb1[i] = b1[idx]
        #
        #     if bnorm is not None:
        #         nrunning_mean[i] = bnorm.running_mean[idx]
        #         nrunning_var[i] = bnorm.running_var[idx]
        #         if bnorm.affine:
        #             nweight[i] = bnorm.weight.data[idx]
        #             nbias[i] = bnorm.bias.data[idx]
        #         bnorm.num_features = new_width
        #
        #     if not random_init:
        #         for idx, d in tracking.items():
        #             for item in d:
        #                 nw2[item].div_(len(d))
        #
        #     w2.transpose_(0, 1)
        #     nw2.transpose_(0, 1)
        #
        #     m1.out_channels = new_width
        #     m2.in_channels = new_width
        #
        #     if noise:
        #         noise = np.random.normal(scale=5e-2 * nw1.std(),
        #                                  size=list(nw1.size()))
        #         nw1 += th.FloatTensor(noise).type_as(nw1)
        #
        #     m1.weight.data = nw1
        #
        #     if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
        #         if w1.dim() == 4:
        #             m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor**2)
        #             m2.in_features = new_width*factor**2
        #         elif w2.dim() == 5:
        #             m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor)
        #             m2.in_features = new_width*factor
        #     else:
        #         m2.weight.data = nw2
        #
        #     m1.bias.data = nb1
        #
        #     if bnorm is not None:
        #         bnorm.running_var = nrunning_var
        #         bnorm.running_mean = nrunning_mean
        #         if bnorm.affine:
        #             bnorm.weight.data = nweight
        #             bnorm.bias.data = nbias
        #     return m1, m2, bnorm
        #
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
