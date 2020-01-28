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

import sys
import torch
from torch.nn.parameter import Parameter

from src import n2n


"""
Make only the (conv, FC) layer parameters sparse 
- Match other layers' parameters when reconfiguring network
- Only work for the flattened networks
"""


def makeSparse(optimizer, model, threshold, is_gating=False, reconf=False):
    print("[INFO] Force the sparse filters to zero...")
    dense_chs, chs_temp, idx = {}, {}, 0
    # alternative List to find the layers by name and not the stupid index of module_list
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

    i = -1
    for name, param in model.named_parameters():
        i = i + 1
        name = altList[i]
        dims = list(param.shape)
        if (('conv' in name) or ('fc' in name)) and ('weight' in name):

            with torch.no_grad():
                param = torch.where(param < threshold, torch.tensor(0.).cuda(), param)

            dense_in_chs, dense_out_chs = [], []
            # param din ==4 -> param is for conv Layer
            if param.dim() == 4:
                # Forcing sparse input channels to zero
                for c in range(dims[1]):
                    if param[:, c, :, :].abs().max() > 0:
                        dense_in_chs.append(c)

                # Forcing sparse output channels to zero
                for c in range(dims[0]):
                    if param[c, :, :, :].abs().max() > 0:
                        dense_out_chs.append(c)

            # Forcing input channels of FC layer to zero
            elif param.dim() == 2:
                # Last FC layers (fc, fc3): Remove only the input neurons
                for c in range(dims[1]):
                    if param[:, c].abs().max() > 0:
                        dense_in_chs.append(c)
                dense_out_chs = [c for c in range(dims[0])]

            chs_temp[idx] = {'name': name, 'in_chs': dense_in_chs, 'out_chs': dense_out_chs}
            idx += 1
            dense_chs[name] = {'in_chs': dense_in_chs, 'out_chs': dense_out_chs, 'idx': idx}

            # print the inter-layer tensor dim [out_ch, in_ch, feature_h, feature_w]
            if reconf:
                print("\n\n Reconf: ")
                if 'fc' in name:
                    print("[{}]: [{}, {}]".format(name,
                                                  len(dense_chs[name]['out_chs']),
                                                  len(dense_chs[name]['in_chs']),
                                                  ))
                else:
                    print("[{}]: [{}, {}, {}, {}]".format(name,
                                                          len(dense_chs[name]['out_chs']),
                                                          len(dense_chs[name]['in_chs']),
                                                          param.shape[2],
                                                          param.shape[3],
                                                          ))
    """
    Inter-layer channel is_gating
    - Union: Maintain all dense channels on the shared nodes (No indexing)
    - Individual: Add gating layers >> Layers at the shared node skip more computation
    """
    # get the residual Path of Resnet
    stagesI, stagesO = model.getResidualPath()
    ch_maps = []

    # Within a residual branch >> Union of adjacent pairs
    # get the Layers that share the same node
    adj_lyrs = model.getShareSameNodeLayers()
    # print(adj_lyrs)
    for adj_lyr in adj_lyrs:
        # if i exists that is in adj_lyr and this i is not in dense_chs
        if any(i for i in adj_lyr if i not in dense_chs):
            """ not doing anything """
        else:
            # print("\n> Adj_lyr: ", adj_lyr)
            for idx in range(len(adj_lyr) - 1):
                edge = list(set().union(dense_chs[adj_lyr[idx]]['out_chs'],
                                        dense_chs[adj_lyr[idx + 1]]['in_chs']))
                # print("\n>Edge: ", edge)
                dense_chs[adj_lyr[idx]]['out_chs'] = edge
                dense_chs[adj_lyr[idx + 1]]['in_chs'] = edge
    # for name in dense_chs:
    # print("1: [{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))

    for idx in range(len(stagesI)):
        # print("\n> IDX: ", idx)
        edges = []
        # Find union of the channels sharing the same node
        for lyr_name in stagesI[idx]:
            # print("\nLyr_name: ", lyr_name)
            if lyr_name in dense_chs:
                edges = list(set().union(edges, dense_chs[lyr_name]['in_chs']))
        for lyr_name in stagesO[idx]:
            # print("\nLyr_name: ", lyr_name)
            if lyr_name in dense_chs:
                edges = list(set().union(edges, dense_chs[lyr_name]['out_chs']))
        # Maintain the dense channels at the shared node
        for lyr_name in stagesI[idx]:
            if lyr_name in dense_chs:
                # print ("Input_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['in_chs']), len(edges)))
                dense_chs[lyr_name]['in_chs'] = edges
        for lyr_name in stagesO[idx]:
            if lyr_name in dense_chs:
                # print ("Output_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['out_chs']), len(edges)))
                dense_chs[lyr_name]['out_chs'] = edges
    # for name in dense_chs:
    #   print("2: [{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))

    return dense_chs, None


"""
Generate a new dense network model
- Rearrange/remove channels from filters
- Rearrange/remove the channels of non-convolution layers
- Remove the dead (all zero channels) layers
- Manage optimization/momentum/buffer parameters
"""


def genDenseModel(model, dense_chs, optimizer, dataset):
    print("[INFO] Squeezing the sparse model to dense one...")

    # Sanity check
    # for layer in dense_chs:
    #     print("==> [{}]: {},{}".format(layer, len(dense_chs[layer]['in_chs']), len(dense_chs[layer]['out_chs'])))

    # List of layers to remove
    rm_list = []
    altList = []
    for name, param in model.named_parameters():
        # print("\nName: {}", name)
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
        else:
            assert True, "Hier fehlt was!! "
    # print("\n> altList: ", altList)
    i = -1
    # print("\nParam: ", paramList)
    # print("==================")
    # for key in optimizer.state:
    #    print("==> {}, {}, {}".format(key, type(key), optimizer.state[key]))
    # for name, param in model.named_parameters():
    for name, param in model.named_parameters():
        i = i + 1
        name = altList[i]
        # print("\nName: ", name)
        # Get Momentum parameters to adjust
        mom_param = optimizer.state[param]['momentum_buffer']

        # Change parameters of neural computing layers (Conv, FC)
        if (('conv' in name) or ('fc' in name)) and ('weight' in name):

            dims = list(param.shape)

            dense_in_ch_idxs = dense_chs[name]['in_chs']
            dense_out_ch_idxs = dense_chs[name]['out_chs']
            num_in_ch, num_out_ch = len(dense_in_ch_idxs), len(dense_out_ch_idxs)

            # print("===> Dense inchs: [{}], outchs: [{}]".format(num_in_ch, num_out_ch))

            # Enlist layers with zero channels for removal
            if num_in_ch == 0 or num_out_ch == 0:
                print("\n RM: ", name)
                rm_list.append(name)
            else:
                # Generate a new dense tensor and replace (Convolution layer)
                if len(dims) == 4:
                    new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()
                    new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()

                    for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                        for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                            with torch.no_grad():
                                new_param[out_idx, in_idx, :, :] = param[out_ch, in_ch, :, :]
                                new_mom_param[out_idx, in_idx, :, :] = mom_param[out_ch, in_ch, :, :]

                # Generate a new dense tensor and replace (FC layer)
                elif len(dims) == 2:
                    new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()
                    new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()

                    for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                        with torch.no_grad():
                            new_param[:, in_idx] = param[:, in_ch]
                            new_mom_param[:, in_idx] = mom_param[:, in_ch]
                else:
                    assert True, "Wrong tensor dimension: {} at layer {}".format(dims, name)

                param.data = new_param
                optimizer.state[param]['momentum_buffer'].data = new_mom_param

                # print("[{}]: {} >> {}".format(name, dims, list(new_param.shape)))

        # Change parameters of non-neural computing layers (BN, biases)
        else:
            # print("\n>Name: ", name)
            w_name = name.replace('bias', 'weight').replace('bn', 'conv')
            # print("\n>WName: ", w_name)
            dense_out_ch_idxs = dense_chs[w_name]['out_chs']
            num_out_ch = len(dense_out_ch_idxs)

            new_param = Parameter(torch.Tensor(num_out_ch)).cuda()
            new_mom_param = Parameter(torch.Tensor(num_out_ch)).cuda()

            for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                with torch.no_grad():
                    new_param[out_idx] = param[out_ch]
                    new_mom_param[out_idx] = mom_param[out_ch]

            param.data = new_param
            optimizer.state[param]['momentum_buffer'].data = new_mom_param

            # print("[{}]: {} >> {}".format(name, dims[0], num_out_ch))

    # print(model)
    # Change moving_mean and moving_var of BN
    for name, buf in model.named_buffers():
        # print("\nBuffer Name: ", name)
        if 'running_mean' in name or 'running_var' in name:
            i = int(name.split('.')[1])
            w_name = 'module.conv' + str(int((i + 1) / 2)) + '.weight'
            # print("\nW_name2: ", w_name)
            dense_out_ch_idxs = dense_chs[w_name]['out_chs']
            num_out_ch = len(dense_out_ch_idxs)
            new_buf = Parameter(torch.Tensor(num_out_ch)).cuda()

            for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                with torch.no_grad():
                    new_buf[out_idx] = buf[out_ch]
            buf.data = new_buf

    """
    Remove layers (Only applicable to ResNet-like networks)
    - Remove model parameters
    - Remove parameters/states in optimizer
    """

    def getLayerIdx(lyr_name):
        if 'conv' in lyr_name:
            conv_id = dense_chs[lyr_name + '.weight']['idx']
            return [3 * (conv_id - 1)], [lyr_name + '.weight']
        elif 'bn' in lyr_name:
            conv_name = lyr_name.replace('bn', 'conv')
            conv_id = dense_chs[conv_name + '.weight']['idx']
            return [3 * conv_id - 1, 3 * conv_id - 2], [lyr_name + '.bias', lyr_name + '.weight']

    if len(rm_list) > 0:
        print("\nRM RM\n")
        rm_lyrs = []
        for name in rm_list:
            print("\n>Name: ", name)
            rm_lyr = n2n.getRmLayers(name, model)
            if any(i for i in rm_lyr if i not in rm_lyrs):
                rm_lyrs.extend(rm_lyr)

        # Remove model parameters
        for rm_lyr in rm_lyrs:
            model.del_param_in_flat_arch(rm_lyr)

        idxs, rm_params = [], []
        for rm_lyr in rm_lyrs:
            idx, rm_param = getLayerIdx(rm_lyr)
            idxs.extend(idx)
            rm_params.extend(rm_param)

        # Remove optimizer states
        for name, param in model.named_parameters():
            for rm_param in rm_params:
                if name == rm_param:
                    del optimizer.state[param]
                    print("\n Del", name)
        # Sanity check: Print out optimizer parameters before change
        print("[INFO] ==== Size of parameter group (Before)")
        for g in optimizer.param_groups:
            for idx, g2 in enumerate(g['params']):
                print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

        # Remove optimizer parameters
        # Adjuster: Absolute parameter location changes after each removal
        for idx_adjuster, idx in enumerate(sorted(idxs)):
            del optimizer.param_groups[0]['params'][idx - idx_adjuster]
            print("\n Del", name)

    # Sanity check => Print out optimizer parameters after change
    print("[INFO] ==== Size of parameter group (After)")
    for g in optimizer.param_groups:
        for idx, g2 in enumerate(g['params']):
            print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

# Sanity check => Check the changed parameters
# for name, param in model.named_parameters():
#  print("===>>> [{}]: {}".format(name, list(param.shape)))

# Sanity check => Check the changed buffers
# for name, param in model.named_parameters():
#  print("===<<< [{}]: {}".format(name, optimizer.state[param]['momentum_buffer'].shape))



def genDense(model, optimizer, dataset):
    print("[INFO] Squeezing the sparse model to dense one...")

    # Sanity check
    # for layer in dense_chs:
    #     print("==> [{}]: {},{}".format(layer, len(dense_chs[layer]['in_chs']), len(dense_chs[layer]['out_chs'])))

    # List of layers to remove
    rm_list = []
    altList = []
    paramList = []
    for name, param in model.named_parameters():
        # print("\nName: {}", name)
        paramList.append(param)
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
        else:
            assert True, "Hier fehlt was!! "
    print("\n> altList: ", altList)
    i = -1
    # print("\nParam: ", paramList)
    # print("==================")
    # for key in optimizer.state:
    #    print("==> {}, {}, {}".format(key, type(key), optimizer.state[key]))
    # for name, param in model.named_parameters():
    for i in range(0, len(altList)):
        name = altList[i]
        param = paramList[i]
        print("\nName: ", name)
        # Get Momentum parameters to adjust
        mom_param = optimizer.state[param]['momentum_buffer']
