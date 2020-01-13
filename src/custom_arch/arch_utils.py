"""
 Copyright 2019 Sangkug Lym
 Copyright 2019 The University of Texas at Austin

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os

from torch import nn


class layerUtil:
    def __init__(self, model, dense_chs, num_classes):
        self.setModel(model, dense_chs)
        self.num_classes = num_classes

    @classmethod
    def setModel(cls, model, dense_chs):
        cls.model = model
        cls.dense_chs = dense_chs

    @classmethod
    def getModuleDef(cls, module):
        if isinstance(module, nn.Conv2d): return cls.convLayer(module)
        elif isinstance(module, nn.BatchNorm2d): return cls.bnLayer(module)
        elif isinstance(module, nn.Linear): return cls.fcLayer(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d): return cls.avgPool(module)

    @classmethod
    def convLayer(cls, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                dims = list(param.shape)
                in_chs = str(dims[1])
                out_chs = str(dims[0])
                kernel_size = str(module.kernel_size)
                stride = str(module.stride)
                padding = str(module.padding)
                bias = module.bias if module.bias != None else True

                return '\t\tlayer = nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={})\n'.format(
          name, in_chs, out_chs, kernel_size, stride, padding, bias)

    @classmethod
    def bnLayer(self, module):
        for name, param in module.named_parameters():
            dims = list(param.shape)
            out_chs = str(dims[0])
            return '\t\tlayer = nn.BatchNorm2d({})\n'.format(name, out_chs)

    def fcLayer(self, module):

        return '\t\tlayer = nn.Linear(16, num_classes)\n'.format(self.num_classes)

    def avgPool(self, module):
        return '\t\tlayer = nn.AdaptiveAvgPool2d((1, 1))\n'

        