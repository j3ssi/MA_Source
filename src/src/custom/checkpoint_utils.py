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
import torch.optim as optim
import numpy as np
import torch.nn as nn

import heapq
import n2n
from .rm_layers import getRmLayers

# Packages to calculate inference cost
from src.src.scripts.feature_size_cifar import cifar_feature_size, imagenet_feature_size

sys.path.append('..')

WORD_SIZE = 4
MFLOPS = 1000000 / 2

""" Return 1D list of weights for the target layer
"""


def _getFilterData(model, target_lyr):
    fil_data = {}
    for i in range(0, len(model.modle_list) - 1):
        if isinstance(model.module_list, nn.Conv2d):
            if target_lyr == model.module_list:
                param = model.module_list[i].weight
                dims = list(param.shape)
                chs = []
                for out_ch in range(dims[0]):
                    chs.append(param.data[out_ch, :, :, :].numpy().flatten())
                fil_data[i] = chs
    return fil_data


""" Return
1. All layers' sparsity heat-map of filters (input channels / output channels)
2. Output channel sparsity by epoch
"""


def _getConvStructSparsity(model, threshold, arch, dataset):
    conv_struct_density = {}
    conv_rand_density = {}
    sparse_bi_map = {}
    sparse_val_map = {}
    conv_id = 0
    model_size = 0
    acc_inf_cost = 0

    if dataset == 'imagenet':
        fmap = imagenet_feature_size[arch]
    else:
        fmap = cifar_feature_size[arch]

    filter_size = 0
    channel_map = []
    tot_weights = 0
    for i in range(0, len(model.module_list) - 1):
        layer = []
        param = model.module_list[i].weight
        dims = list(param.shape)
        if isinstance(model.module_list[i], nn.Conv2d):
            channel_map = np.zeros([dims[1], dims[0]])
            filter_size = dims[2] * dims[3]
            for in_ch in range(dims[1]):
                fil_row = []
                for out_ch in range(dims[0]):
                    fil = param.data.numpy()[out_ch, in_ch, :, :]
                    fil_max = np.absolute(fil).max()
                    fil_row.append(fil_max)
                    # if fil_max > threshold:
                    if fil_max > 0.:
                        channel_map[in_ch, out_ch] = 1
                layer.append(fil_row)

        elif isinstance(model.module_list[i], nn.Linear):
            channel_map = np.zeros([dims[1], dims[0]])
            filter_size = 1
            for in_ch in range(dims[1]):
                fil_row = []
                for out_ch in range(dims[0]):
                    fil = param.data.numpy()[out_ch, in_ch]
                    fil_max = np.absolute(fil)
                    fil_row.append(fil_max)
                    # if fil_max > threshold:
                    if fil_max > 0.:
                        channel_map[in_ch, out_ch] = 1
                layer.append(fil_row)

        # ratio of non_zero weights
        weights = param.data.numpy()
        weight_density = float(weights[weights >= threshold].size) / weights.size

        tot_weights += weights[weights >= threshold].size

        sparse_val_map[conv_id] = np.array(layer)
        sparse_bi_map[conv_id] = channel_map

        rows = channel_map.max(axis=1)  # in_channels
        cols = channel_map.max(axis=0)  # out_channels

        num_dense_out_ch = float(np.count_nonzero(cols))
        num_dense_in_ch = float(np.count_nonzero(rows))

        out_density = num_dense_out_ch / len(cols)
        in_density = num_dense_in_ch / len(rows)

        conv_struct_density[conv_id] = {'in_ch': in_density, 'out_ch': out_density}
        conv_rand_density[conv_id] = weight_density
        # print("{}: {}".format(name, weight_density))

        model_size += num_dense_out_ch * num_dense_in_ch * filter_size  # Add filters
        model_size += num_dense_out_ch  # Add bias

        # Calculate inference cost = (CRS)(K)(NPQ)
        # fmap_name = name.split('module.')[1].split('.weight')[0]
        if isinstance(model.module_list[i], nn.Conv2d):
            inf_cost = (num_dense_in_ch * dims[2] * dims[3]) * num_dense_out_ch * (fmap[i][1] ** 2)
        else:
            inf_cost = (num_dense_in_ch * num_dense_out_ch)
        # print("{}, {}, {}".format(fmap_name, num_dense_in_ch, num_dense_out_ch))

        conv_id += 1
        acc_inf_cost += inf_cost

    print("tot_weights:{}".format(tot_weights))

    return (sparse_bi_map, sparse_val_map, conv_id, conv_struct_density,
            conv_rand_density, (model_size * WORD_SIZE), acc_inf_cost / MFLOPS)


"""
Make only the (conv, FC) layer parameters sparse 
- Match other layers' parameters when reconfiguring network
- Only work for the flattened networks
"""


def _makeSparse(model, threshold, is_gating=False, reconf=True):
    print("[INFO] Force the sparse filters to zero...")
    dense_chs, chs_temp, idx = {}, {}, 0
    altList = []
    for name, param in model.named_parameters():
        i = int(name.split('.')[1])
        if i % 2 == 0:
            altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')

        if (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
        elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")

        if (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
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
                # FC layer in the middle remove their output neurons
                if any(i for i in ['fc1', 'fc2'] if i in name):
                    for c in range(dims[0]):
                        if param[c, :].abs().max() > 0:
                            dense_out_chs.append(c)
                else:
                    # [fc, fc3] output channels (class probabilities) are all dense
                    dense_out_chs = [c for c in range(dims[0])]

            chs_temp[idx] = {'name': name, 'in_chs': dense_in_chs, 'out_chs': dense_out_chs}
            idx += 1
            dense_chs[name] = {'in_chs': dense_in_chs, 'out_chs': dense_out_chs, 'idx': idx}

            # print the inter-layer tensor dim [out_ch, in_ch, feature_h, feature_w]
            if not reconf:
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
    stages = n2n.getResidualPath(model)
    ch_maps = []

    # Within a residual branch >> Union of adjacent pairs
    adj_lyrs = n2n.getShareSameNodeLayers(model)
    # print(adj_lyrs)
    for adj_lyr in adj_lyrs:
        if any(i for i in adj_lyr if i not in dense_chs):
            """ not doing anything """
        else:
            for idx in range(len(adj_lyr) - 1):
                edge = list(set().union(dense_chs[adj_lyr[idx]]['out_chs'],
                                        dense_chs[adj_lyr[idx + 1]]['in_chs']))
                dense_chs[adj_lyr[idx]]['out_chs'] = edge
                dense_chs[adj_lyr[idx + 1]]['in_chs'] = edge

    for idx in range(len(stages) - 1):
        edges = []
        # Find union of the channels sharing the same node
        for lyr_name in stages[idx]['i']:
            if lyr_name in dense_chs:
                edges = list(set().union(edges, dense_chs[lyr_name]['in_chs']))
        for lyr_name in stages[idx]['o']:
            if lyr_name in dense_chs:
                edges = list(set().union(edges, dense_chs[lyr_name]['out_chs']))
        # Maintain the dense channels at the shared node
        for lyr_name in stages[idx]['i']:
            if lyr_name in dense_chs:
                # print ("Input_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['in_chs']), len(edges)))
                dense_chs[lyr_name]['in_chs'] = edges
        for lyr_name in stages[idx]['o']:
            if lyr_name in dense_chs:
                # print ("Output_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['out_chs']), len(edges)))
                dense_chs[lyr_name]['out_chs'] = edges

    for name in dense_chs:
        print ("[{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))
    return dense_chs, None


"""
Generate a new dense network model
- Rearrange/remove channels from filters
- Rearrange/remove the channels of non-convolution layers
- Remove the dead (all zero channels) layers
- Manage optimization/momentum/buffer parameters
"""


def _genDenseModel(model, dense_chs, optimizer, dataset):
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

        if (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
        elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")

        if (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
            altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
        elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
            altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")

    i = -1

    # print("==================")
    # for key in optimizer.state:
    #    print("==> {}, {}, {}".format(key, type(key), optimizer.state[key]))
    for name, param in model.named_parameters():
        i = i + 1
        name = altList[i]

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

                    if ('fc1' in name) or ('fc2' in name):
                        for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                            for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                                with torch.no_grad():
                                    new_param[out_idx, in_idx] = param[out_ch, in_ch]
                                    new_mom_param[out_idx, in_idx] = mom_param[out_ch, in_ch]
                    else:
                        for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                            with torch.no_grad():
                                new_param[:, in_idx] = param[:, in_ch]
                                new_mom_param[:, in_idx] = mom_param[:, in_ch]
                else:
                    assert True, "Wrong tensor dimension: {} at layer {}".format(dims, name)

                param.data = new_param
                optimizer.state[param]['momentum_buffer'].data = new_mom_param

                print("[{}]: {} >> {}".format(name, dims, list(new_param.shape)))

        # Change parameters of non-neural computing layers (BN, biases)
        else:
            w_name = name.replace('bias', 'weight').replace('bn', 'conv')
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

            #print("[{}]: {} >> {}".format(name, dims[0], num_out_ch))

    print(model)
    # Change moving_mean and moving_var of BN
    for name, buf in model.named_buffers():
        if 'running_mean' in name or 'running_var' in name:
            i = int(name.split('.')[1])
            w_name = 'module.conv' + str(int((i + 1) / 2)) + '.weight'
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
    print("TODO: hier noch nicht angepasst")
    if len(rm_list) > 0:
        rm_lyrs = []
        for name in rm_list:
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
        # print ("[INFO] ==== Size of parameter group (Before)")
        # for g in optimizer.param_groups:
        #  for idx, g2 in enumerate(g['params']):
        #    print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

        # Remove optimizer parameters
        # Adjuster: Absolute parameter location changes after each removal
        for idx_adjuster, idx in enumerate(sorted(idxs)):
            del optimizer.param_groups[0]['params'][idx - idx_adjuster]
            print("\n Del", name)

    # Sanity check => Print out optimizer parameters after change
    # print ("[INFO] ==== Size of parameter group (After)")
    # for g in optimizer.param_groups:
    #  for idx, g2 in enumerate(g['params']):
    #    print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

# Sanity check => Check the changed parameters
# for name, param in model.named_parameters():
#  print("===>>> [{}]: {}".format(name, list(param.shape)))

# Sanity check => Check the changed buffers
# for name, param in model.named_parameters():
#  print("===<<< [{}]: {}".format(name, optimizer.state[param]['momentum_buffer'].shape))
