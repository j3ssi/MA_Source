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
    def __init__(self, model, dense_chs):
        self.setModel(model, dense_chs)

    @classmethod
    def setModel(cls, model, dense_chs):
        cls.model = model
        cls.dense_chs = dense_chs

    def getModuleDef(cls, module):
        if isinstance(module, nn.Conv2d): return cls.convLayer(module)
        elif isinstance(module, nn.BatchNorm2d): return cls.bnLayer(module)
        elif isinstance(module, nn.Linear): return cls.fcLayer(module)
        elif isinstance(module, nn.AdaptiveAvgPool2d): return cls.avgPool(module)

    def convLayer(cls, module):
        pass

    def bnLayer(self, module):
        pass

    def fcLayer(self, module):
        pass

    def avgPool(self, module):
        pass
        