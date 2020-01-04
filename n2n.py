import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class N2N(nn.Module):

    def __init__(self, num_classes):
        super(N2N, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn1 = nn.BatchNorm2d(16)

        # 1
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn3 = nn.BatchNorm2d(16)

        # 2
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn5 = nn.BatchNorm2d(16)

        # 3
        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn7 = nn.BatchNorm2d(16)

        # 4
        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn8 = nn.BatchNorm2d(16)
        self.conv9 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn9 = nn.BatchNorm2d(16)

        # 5
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        _x = self.relu(x)
        i = 2
        while i > 0:
            convStr = 'conv' + str(i)
            names = self.__dict__.__getitem__('_modules')

            if convStr not in names:
                x = self.avgpool(_x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
            # find the module with name convStr
            for name, module in self.named_modules():
                if name == convStr:
                    try:
                        x = module.forward(_x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")

            bnStr = 'bn' + str(i)
            for name, module in self.named_modules():
                if name == bnStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")
            x = self.relu(x)
            i = i + 1
            convStr = 'conv' + str(i)
            for name, module in self.named_modules():
                if name == convStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")

            bnStr = 'bn' + str(i)

            for name, module in self.named_modules():
                if name == bnStr:
                    try:
                        x = module.forward(x)
                        break
                    except RuntimeError:
                        print("\n \n Oops!!! \n \n \n")
            try:
                _x = _x + x
            except RuntimeError:
                print("\n \n Oops!!  \n \n \n")

            _x = self.relu(_x)
            i = i + 1

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def deeper(self, model, positions):
    #each pos is the position in which the layer sholud be duplicated to make the cnn deeper
    for pos in positions:
        posStr = 'conv' + str(pos)
        names = self.__dict__.__getitem__('_modules')
        j = pos + 1
        for name, module in model.named_parameters():
            if posStr in name:
                i = name.index(posStr)
                conv1 = module[i]
                conv2 = conv1.clone()
                convStr3 = 'conv' + j
                if convStr3 not in names:
                    posStr = 'conv' + str(j)
                    self.__dict__[posStr] = conv2
                    print(self.__dict__.__getitem__('_modules'))
                    return model
                else:
                    conv3 = module[i+1]
                    posStr = 'conv' + str(pos+1)
                    self.__dict__[posStr] = conv2
                    j=j+1
                    break
        while j>0:
            convStr = 'conv' + str(j)
            if convStr not in names:
                return model
            else:
                conv1 = module[i]
                conv2 = conv1.clone()
                convStr3 = 'conv' + j
                if convStr3 not in names:
                    return model
                else:
                    conv3 = module[i + 1]
                    posStr = 'conv' + str(pos + 1)
                    self.__dict__[posStr] = conv2
                    j = j + 1
        for name, module in model.named_parameters():
            posStr = 'conv'+ str(pos+i)
            #    posStr1 = 'conv' + posModel
             #   name[posModel] = posStr1
              #  model[posModel + 1] = model[posModel]
               # model[posModel] = conv2
            #else:
             #   print(name[posModel])

