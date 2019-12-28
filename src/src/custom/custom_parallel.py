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

import torch.nn as nn
import numpy as np
from Net2Net.net2net import deeper, wider


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)



    """ Remove sparsified module parameter from the network model
    # rm_name: name of module to remove
    """

    def del_param_in_flat_arch(self, rm_name):
        # We remove an entire layer holding the delete target parameters
        rm_module = rm_name.split('.')
        module = self._modules[rm_module[0]]
        if module._modules[rm_module[1]] != None:
            print("[INFO] Removing parameters/buffers in module [{}]".format(rm_module[0] + '.' + rm_module[1]))
            del module._modules[rm_module[1]]

    def net2net_wider(self, args):
        self.conv1, self.conv2, _ = wider(self.conv1, self.conv2, 12,
                                          self.bn1, noise=args.noise)
        self.conv2, self.conv3, _ = wider(self.conv2, self.conv3, 24,
                                          self.bn2, noise=args.noise)
        self.conv3, self.fc1, _ = wider(self.conv3, self.fc1, 48,
                                        self.bn3, noise=args.noise)
        print(self)

    def net2net_deeper(self, args):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s
        print(self)

    def net2net_deeper_nononline(self, args):
        s = deeper(self.conv1, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s
        print(self)

    def define_wider(self, args):
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 3 * 3, 10)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(),
                                   nn.Conv2d(12, 12, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(24),
                                   nn.ReLU(),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 3 * 3, 10)
        print(self)


def net2net_deeper_recursive(model):
    """
    Apply deeper operator recursively any conv layer.
    """
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            s = deeper(module, nn.ReLU, bnorm_flag=False)
            model._modules[name] = s
        elif isinstance(module, nn.Sequential):
            module = net2net_deeper_recursive(module)
            model._modules[name] = module
    return model
