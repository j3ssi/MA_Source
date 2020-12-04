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
from torch import nn
from torch.nn.parameter import Parameter

from src import n2n

"""
Make only the (conv, FC) layer parameters sparse 
- Match other layers' parameters when reconfiguring network
- Only work for the flattened networks
"""


def makeSparse(optimizer, model, threshold, reconf=True):
    print("[INFO] Force the sparse filters to zero...")
    dense_chs, chs_temp, idx = {}, {}, 0
    # alternative List to find the layers by name and not the stupid index of module_list
    altList = []
    for index in range( len( model.module_list ) ):
        if isinstance(model.module_list[index], nn.Conv2d):
            altList.append((index,None))
        elif isinstance(model.module_list[index],nn.Sequential):
            moduleX = model.module_list[index]
            for i in range( len( moduleX ) ):
                if isinstance(moduleX[i], nn.Conv2d):
                    altList.append((index,i))
        elif isinstance(model.module_list[index], nn.Linear):
            altList.append((index,None))

    # print(f'altList: {altList}')
    i = -1
    for i,j in altList:
        module = model.module_list[i]
        if j is not None:
            module = module[j]
        param = module.weight

        dims = list(param.shape)
        # print(f'name: {name}; dims {dims}')

        with torch.no_grad():
            param = torch.where(param < threshold, torch.tensor(0.).cuda(), param)

        dense_in_chs, dense_out_chs = [], []
        # param din ==4 -> param is for conv Layer
        if param.dim() == 4:
            if i == 0:
                dense_in_chs.append(0)
                dense_in_chs.append(1)
                dense_in_chs.append(2)
            # Forcing sparse input channels to zero
            else:
                for c in range(dims[1]):
                    if param[:, c, :, :].abs().max() > 0:
                        dense_in_chs.append(c)
                        # print(f'Dense In Ch. {c}')

            # Forcing sparse output channels to zero
            for c in range(dims[0]):
                if param[c, :, :, :].abs().max() > 0:
                    dense_out_chs.append(c)
                    # print(f'Dense Out Ch. {c}')

        # Forcing input channels of FC layer to zero
        elif param.dim() == 2:
            # Last FC layers (fc, fc3): Remove only the input neurons
            for c in range(dims[1]):
                if param[:, c].abs().max() > 0:
                    dense_in_chs.append(c)
            dense_out_chs = [c for c in range(dims[0])]

        chs_temp[idx] = {'i': i, 'j': j, 'in_chs': dense_in_chs, 'out_chs': dense_out_chs}
        idx += 1
        dense_chs[(i,j)] = {'in_chs': dense_in_chs, 'out_chs': dense_out_chs, 'idx': idx}

        # print the inter-layer tensor dim [out_ch, in_ch, feature_h, feature_w]
        if reconf:
            print("\n\n Reconf: ")
            if isinstance(module, nn.Linear):
                print("[{}]: [{}, {}]".format((i,j),
                                              len(dense_chs[(i,j)]['out_chs']),
                                              len(dense_chs[(i,j)]['in_chs']),
                                              ))
            else:
                print("[{}]: [{}, {}, {}, {}]".format((i,j),
                                                      len(dense_chs[(i,j)]['out_chs']),
                                                      len(dense_chs[(i,j)]['in_chs']),
                                                      param.shape[2],
                                                      param.shape[3],
                                                      ))
    """
    Inter-layer channel is_gating
    - Union: Maintain all dense channels on the shared nodes (No indexing)
    - Individual: Add gating layers >> Layers at the shared node skip more computation
    """
    # get the residual Path of Resnet
    # stagesI, stagesO = model.getResidualPath()
    ch_maps = []

    # Within a residual branch >> Union of adjacent pairs
    # get the Layers that share the same node
    # adj_lyrs = model.getShareSameNodeLayers()
    # print(adj_lyrs)
    i = 0
    while i < len( model.module_list ) :
        adj_lyrs = []
        module = model.module_list[i]
        print(f'i: {i}')
        if isinstance(module, nn.Sequential):
            size1 = module[0].out_channels
            for j in range(1, len ( module ) ):
                if isinstance(module[j], nn.Conv2d):
                    size0 = module[j].in_channels

                    if (i,j) in altList and size0 == size1 and (i,0) not in adj_lyrs:
                        print(f'0: (i,j): ({i},{j})')
                        adj_lyrs.append((i,0))
                        adj_lyrs.append((i,j))
                    elif (i,j) in altList and size0 == size1:
                        print(f'(i,j): ({i},{j})')
                        adj_lyrs.append((i,0))
                        adj_lyrs.append((i,j))


                    size1 = module[j].out_channels

        for adj_lyr in adj_lyrs:
        #if i exists that is in adj_lyr and this i is not in dense_chs
            if any(i for i in adj_lyr if i not in dense_chs):
                """ not doing anything """
            else:
                # print("\n> Adj_lyr: ", adj_lyr)
                for idx in range(len(adj_lyr) - 1):
                    edge = list(set().union(dense_chs[adj_lyr[idx]]['out_chs'],
                                            dense_chs[adj_lyr[idx + 1]]['in_chs']))
                    print("\n>Edge: ", edge)
                    dense_chs[adj_lyr[idx]]['out_chs'] = edge
                    dense_chs[adj_lyr[idx + 1]]['in_chs'] = edge
        i += 1

    for name in dense_chs:
        print("1: [{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))
    width = model.module_list[0].weight.size(0)
    # print(f'Start width: {width}')
    idx = 0
    new_edges = []
    listOTmp = None
    while idx < len( model.module_list ):
        # print("\n> IDX: ", idx)

        edges = []
        edges = list(set().union(edges, new_edges))
        new_edges = []
        # module = model.module_list[idx]
        listI = []
        listO = []
        if listOTmp is not None:
            listO.append(listOTmp)
        listOTmp = None
        layerWidth = width
        while width == layerWidth:
            module = model.module_list[idx]

            # print(f'Width: {width}; layerWidth: {layerWidth}; idx: {idx}')
            if isinstance(module, nn.Sequential):
                layerWidth = module[0].weight.size(0)
                # print(f'layerWidth: {layerWidth}')
                if (idx,0) in dense_chs and layerWidth == width:
                    edges = list(set().union(edges, dense_chs[(idx,0)]['in_chs']))
                    listI.append((idx,0))
                    # print(f'Append I: {(idx,0)}')
                else:
                    width = module[0].weight.size(0)
                    break
                j = len(module)-1
                # print(f'J: {j}')
                while j>0:
                    if isinstance(module[j], nn.Conv2d):
                        if module[j].weight.size(1) == width:
                            edges = list(set().union(edges, dense_chs[(idx,j)]['out_chs']))
                            listO.append((idx,j))
                            # print(f'Append O: {(idx,j)}')
                            j = j-1
                        else:
                            new_edges = list(set().union(edges, dense_chs[(idx,j)]['out_chs']))
                            width = module[j].weight.size(1)
                            listOTmp = (idx,j)
                            break
                    else:
                        j = j - 1
                        # print(f'J: {j}')
                idx += 1

            if isinstance(module, nn.Conv2d):
                layerWidth = module.weight.size(0)
                if (idx,None) in dense_chs:
                    edges = list(set().union(edges, dense_chs[(idx, None)]['out_chs']))
                    listO.append((idx, None))
                    # print(f'Append O : {(idx,None)}')
                    idx += 1
            if isinstance(module, nn.Linear):
                if (idx,None) in dense_chs:
                    edges = list(set().union(edges, dense_chs[(idx, None)]['out_chs']))
                    listI.append((idx,None))
                    # print(f'Append I: {(idx,None)}')
                    idx += 1
                    break
            else:
                idx += 1
        # print(f'listI: {listI}')
        # print(f'listO: {listO}')

        for i,j in listI:
            if (i,j) in dense_chs:
                dense_chs[(i,j)]['in_chs'] = edges
                print(f'listI (i,j): {i},{j}; edges: {edges} ')
        for i, j in listO:
            if (i, j) in dense_chs:
                dense_chs[(i, j)]['out_chs'] = edges
                print(f'listO (i,j): {i},{j}; edges: {edges} ')

    #     # Maintain the dense channels at the shared node
    # for lyr_name in stagesI[idx]:
    #     if lyr_name in dense_chs:
    #             # print ("Input_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['in_chs']), len(edges)))
    #             dense_chs[lyr_name]['in_chs'] = edges
    #     for lyr_name in stagesO[idx]:
    #         if lyr_name in dense_chs:
    #             # print ("Output_ch [{}]: {} => {}".format(lyr_name, len(dense_chs[lyr_name]['out_chs']), len(edges)))
    #             dense_chs[lyr_name]['out_chs'] = edges
    # # for name in dense_chs:
    # #    print("2: [{}]: {}, {}".format(name, dense_chs[name]['in_chs'], dense_chs[name]['out_chs']))

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
    # for name, param in model.named_parameters():
    #     # print("\nName: {}", name)
    #     i = int(name.split('.')[1])
    #     if i % 2 == 0:
    #         altList.append('module.conv' + str(int((i / 2) + 1)) + '.weight')
    #
    #     elif (i % 2 == 1) and ('weight' in name) and (i < (len(model.module_list) - 2)):
    #         altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".weight")
    #     elif (i % 2 == 1) and ('weight' in name) and (i > (len(model.module_list) - 3)):
    #         altList.append('module.fc' + str(int((i + 1) / 2)) + ".weight")
    #
    #     elif (i % 2 == 1) and ('bias' in name) and (i < (len(model.module_list) - 1)):
    #         altList.append('module.bn' + str(int(((i - 1) / 2) + 1)) + ".bias")
    #     elif (i % 2 == 1) and ('bias' in name) and (i > (len(model.module_list) - 2)):
    #         altList.append('module.fc' + str(int((i + 1) / 2)) + ".bias")
    #     else:
    #         assert True, "Hier fehlt was!! "
    # # print("\n> altList: ", altList)
    # i = -1
    # print("\nParam: ", paramList)
    # print("==================")
    # for key in optimizer.state:
    #    print("==> {}, {}, {}".format(key, type(key), optimizer.state[key]))
    for name, param in model.named_parameters():
        i = int(name.split('.')[1])
        j = None
        if len(name.split('.')) ==4:
            j = int(name.split('.')[2])
        # print(f'(i,j): ({i}, {j})')
        print("\nName: ", name)

        name = (i,j)
        # Get Momentum parameters to adjust
        mom_param = optimizer.state[param]['momentum_buffer']
        module = model.module_list[i]
        if j is not None:
            module = module[j]
        print(f'Module: {module}')
        # Change parameters of neural computing layers (Conv, FC)
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            print("\nName: ", name)

            dims = list(param.shape)
            # print(f'dims: {dims}')
            dense_in_ch_idxs = dense_chs[name]['in_chs']
            print(f'in chs: {dense_in_ch_idxs}')
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
                    # print(f'Dims = 4')
                    new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()
                    new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch, dims[2], dims[3])).cuda()
                    # print(f'Sorted: {sorted(dense_out_ch_idxs)}')
                    for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                        for out_idx, out_ch in enumerate(sorted(dense_out_ch_idxs)):
                            with torch.no_grad():
                                new_param[out_idx, in_idx, :, :] = param[out_ch, in_ch, :, :]
                                new_mom_param[out_idx, in_idx, :, :] = mom_param[out_ch, in_ch, :, :]
                        # print(f'in_idx: {in_idx}; out_idx: {out_idx}; in_ch: {in_ch}; out_ch: {out_ch}')

                # Generate a new dense tensor and replace (FC layer)
                elif len(dims) == 2:
                    # print(f'Dims =2')
                    new_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()
                    new_mom_param = Parameter(torch.Tensor(num_out_ch, num_in_ch)).cuda()

                    for in_idx, in_ch in enumerate(sorted(dense_in_ch_idxs)):
                        with torch.no_grad():
                            new_param[:, in_idx] = param[:, in_ch]
                            new_mom_param[:, in_idx] = mom_param[:, in_ch]
                else:
                    assert True, "Wrong tensor dimension: {} at layer {}".format(dims, name)
                # print(f'Übertrage new Param')
                param.data = new_param
                optimizer.state[param]['momentum_buffer'].data = new_mom_param

                # print("[{}]: {} >> {}".format(name, dims, list(new_param.shape)))

        # Change parameters of non-neural computing layers (BN, biases)
        else:

            if j == None:
                w_name = (i-1,j)
            else:
                w_name = (i,j-1)
            print("\nWName: ", w_name)

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

    # print(f'Change moving mean and var of BN')
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
    # rm_list=[ 'module.conv21.weight' ,' module.conv22.weight']
    if len(rm_list) > 0:
        indexList = []
        j = 2
        m = 0
        # print(f'RM List vorher: {rm_list}')

        for rm in rm_list:
            index = int(rm.split('.')[1].split('v')[1])
            # index = (index - 1) * 2
            indexList.append(index)
        # (f'indexList: {indexList}')
        for stage in range(0, model.numOfStages):
            # print(f'Stage: {stage}')
            archNum = model.archNums[stage]

            for block in range(0, len(archNum)):
                # print(f'Block: {block}')
                arch = archNum[block]
                sameBlock = False
                i = 0
                while i < arch:
                    if len(indexList) > 0:
                        if indexList[0] == j and not sameBlock:
                            elem = indexList[0]
                            indexList.remove(elem)
                            # print(f'pop element: {elem}')
                            # print(f'IndexListe: {indexList}')
                            sameBlock = True
                            m = m + 1
                            j = j + 1
                            i = i + 1
                        elif indexList[0] == j and sameBlock:
                            elem = indexList[0]
                            indexList.remove(elem)
                            # print(f'pop element2: {elem}')
                            delete = rm_list[m]
                            # print(f'delete: {delete}')
                            rm_list.remove(delete)
                            j = j + 1
                            i = i + 1
                        else:
                            j = j + 1
                            i = i + 1

                    else:
                        i = arch + 1
        # print(f'RM List nachher: {rm_list}')
        for name in reversed(rm_list):
            # delete module from moduleList
            index = int(name.split('.')[1].split('v')[1])
            index = (index - 1) * 2
            module = model.module_list[index]
            # print("\nModule List Length: ", len(model.module_list))
            model.delete(module, index)
            # print("\nModule List Length After Delete: ", len(model.module_list))
            # i=0
            # for s in range(0, model.numOfStages):
            #     blocks = model.archNums[s]
            #     for b in range(0, model.numOfBlocksinStage):
            #         i = i + blocks[b]
            #         if(index == i):
            #             blocks[b] = blocks[b] - 1

        # # Sanity check: Print out optimizer parameters before change
        # print("[INFO] ==== Size of parameter group (Before)")
        # for g in optimizer.param_groups:
        #     for idx, g2 in enumerate(g['params']):
        #         print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

    # Sanity check => Print out optimizer parameters after change
    # print("[INFO] ==== Size of parameter group (After)")
    # for g in optimizer.param_groups:
    #    for idx, g2 in enumerate(g['params']):
    #        print("idx:{}, param_shape:{}".format(idx, list(g2.shape)))

# Sanity check => Check the changed parameters
# for name, param in model.named_parameters():
#  print("===>>> [{}]: {}".format(name, list(param.shape)))

# Sanity check => Check the changed buffers
# for name
# , param in model.named_parameters():
#  print("===<<< [{}]: {}".format(name, optimizer.state[param]['momentum_buffer'].shape))
