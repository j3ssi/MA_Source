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

import torch
import torch.nn as nn

""" A single global group-lasso regularization coefficient
# 1. Exclude depth-wise separable convolution from regularization
# 2. Exclude first layer's input channel and last layer's output from regularization
# 3. Consider multi-layer classifier

# arch: architecture name
# lasso_penalty: group lasso regularization penalty
"""

def get_group_lasso_global(model):
    lasso_in_ch = []
    lasso_out_ch = []
    altList = []
    for name, param in model.named_parameters():
        i = int(name.split('.')[1])
        if i % 2 == 0:
            altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')

        elif (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
        elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")

        elif (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
        elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
        # print(altList[-1])

    # print("\naltList: ", altList)
    j = -1
    for name, param in model.named_parameters():
        j = j + 1
        name = altList[j]
        # Lasso added to only the neuronal layers
        if ('weight' in name) and any([i for i in ['conv', 'fc'] if i in name]):
            # print("\nName: ", name)
            if param.dim() == 4:

                # Exclude depth-wise convolution layers from regularization
                if 'conv1.' not in name:
                    _in = param.pow(2).sum(dim=[0, 2, 3])
                    lasso_in_ch.append(_in)

                _out = param.pow(2).sum(dim=[1, 2, 3])
                lasso_out_ch.append(_out)
                # print("\n\n>Name,lasso_ch: ", name, " ; ", lasso_in_ch, " ;", lasso_out_ch)
            elif param.dim() == 2:
                lasso_in_ch.append(param.pow(2).sum(dim=[0]))

    _lasso_in_ch = torch.cat(lasso_in_ch).cuda()
    _lasso_out_ch = torch.cat(lasso_out_ch).cuda()
    lasso_penalty_in_ch = _lasso_in_ch.add(1.0e-8).sqrt().sum()
    lasso_penalty_out_ch = _lasso_out_ch.add(1.0e-8).sqrt().sum()
    # print(f'Lasso in: {lasso_penalty_in_ch}')
    # print(f'Lasso out: {lasso_penalty_out_ch}')

    lasso_penalty = lasso_penalty_in_ch + lasso_penalty_out_ch
    return lasso_penalty


""" Number of parameter-based per-group regularization coefficient
# 1. Exclude depth-wise separable convolution from regularization
# 2. Exclude first layer's input channel and last layer's output from regularization
# 3. Consider multi-layer classifier

# arch: architecture name
# lasso_penalty: group lasso regularization penalty
"""


def get_group_lasso_group(model):
    lasso_in_ch = []
    lasso_out_ch = []
    lasso_in_ch_penalty = []
    lasso_out_ch_penalty = []

    altList = []
    for name, param in model.named_parameters():
        i = int(name.split('.')[1])
        if i % 2 == 0:
            altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')

        elif (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
        elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")

        elif (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
        elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
        # print(altList[-1])

    j = -1

    for name, param in model.named_parameters():
        j = j + 1
        name = altList[j]
        # Lasso added to only the neuronal layers
        if ('weight' in name) and any([i for i in ['conv', 'fc'] if i in name]):
            if param.dim() == 4:
                w_num_i_ch = param.shape[0] * param.shape[2] * param.shape[3]
                w_num_o_ch = param.shape[1] * param.shape[2] * param.shape[3]

                # Exclude depth-wise convolution layers from regularization
                if 'conv1.' not in name:
                    _in = param.pow(2).sum(dim=[0, 2, 3])
                    lasso_in_ch.append(_in)
                    penalty_tensor = torch.Tensor(param.shape[1]).cuda()
                    lasso_in_ch_penalty.append(penalty_tensor.new_full([param.shape[1]], w_num_i_ch))

                _out = param.pow(2).sum(dim=[1, 2, 3])
                lasso_out_ch.append(_out)
                penalty_tensor = torch.Tensor(param.shape[0]).cuda()
                lasso_out_ch_penalty.append(penalty_tensor.new_full([param.shape[0]], w_num_o_ch))

            elif param.dim() == 2:
                w_num_i_ch = param.shape[0]
                lasso_in_ch.append(param.pow(2).sum(dim=[0]))
                penalty_tensor = torch.Tensor(param.shape[1]).cuda()
                lasso_in_ch_penalty.append(penalty_tensor.new_full([param.shape[1]], w_num_i_ch))

    _lasso_in_ch = torch.cat(lasso_in_ch).cuda()
    _lasso_out_ch = torch.cat(lasso_out_ch).cuda()
    lasso_penalty_in_ch = _lasso_in_ch.add(1.0e-8).sqrt()
    lasso_penalty_out_ch = _lasso_out_ch.add(1.0e-8).sqrt()

    # Extra penalty using the number of parameters in each group
    lasso_in_ch_penalty = torch.cat(lasso_in_ch_penalty).cuda().sqrt()
    lasso_out_ch_penalty = torch.cat(lasso_out_ch_penalty).cuda().sqrt()
    lasso_penalty_in_ch = lasso_penalty_in_ch.mul(lasso_in_ch_penalty).sum()
    lasso_penalty_out_ch = lasso_penalty_out_ch.mul(lasso_out_ch_penalty).sum()

    lasso_penalty = lasso_penalty_in_ch + lasso_penalty_out_ch
    return lasso_penalty
