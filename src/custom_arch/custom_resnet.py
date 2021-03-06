import os

from src.custom_arch import layerUtil


def _genDenseArchResNet(model, out_dir, dense_chs, chs_map, num_classes):
    ctx = 'import torch.nn as nn\n'
    ctx += 'import math\n'
    ctx += '__all__ = [\'N2N\']\n'
    ctx += 'class N2N(nn.Module):\n'
    ctx += '\tdef __init__(self, num_classes=10):\n'
    ctx += '\t\tsuper(N2N, self).__init__()\n'

    #lyr = layerUtil(model, dense_chs)

    ctx += '\t\tself.module_list = nn.ModuleList()\n'
    altList = []
    paramList = []
    for name, param in model.named_parameters():
        # print("\nName: {}", name)
        i = int(name.split('.')[1])
        paramList.append(param)
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

    print(altList)
    i = -1
    for name, param in model.named_parameters():
        i = i + 1
        name = altList[i]
        if 'conv' in name:
            dims = list(param.shape)
            in_chs = str(dims[1])
            out_chs = str(dims[0])
            #Search for the corresponding Conv Module in Module_list
            k = int(name.split('.')[1].split('v')[1])
            module = model.module_list[(k-1)*2]
            print("\nName module", k)
            kernel_size = str(module.kernel_size)
            stride = str(module.stride)
            padding = str(module.padding)
            bias = module.bias if module.bias != None else True

            ctx += '\t\tlayer = nn.Conv2d({}, {}, kernel_size={}, stride={}, padding={}, bias={})\n'.format(
                name, in_chs, out_chs, kernel_size, stride, padding, bias)
            ctx += '\t\tmodule_list.append(layer)\n'
        elif 'bn' in name and not 'bias' in name:
            dims = list(param.shape)
            out_chs = str(dims[0])
            ctx += '\t\tlayer = nn.BatchNorm2d({})\n'.format(name, out_chs)
            ctx += '\t\tmodule_list.append(layer)\n'
        else:
            print('\nelse: ',name)

    ctx += '\t\tlayer = nn.AdaptiveAvgPool2d((1, 1))\n'
    ctx += '\t\tmodule_list.append(layer)\n'
    ctx += '\t\tlayer = nn.Linear(16, 10)\n'
    ctx += '\t\tmodule_list.append(layer)\n'

    ctx += '\tdef forward(self,x):\n'
    ctx += '\t\todd = False\n'
    ctx += '\t\tfirst = True\n'
    ctx += '\t\tbn = False\n'
    ctx += '\t\t_x = None\n'
    ctx += '\t\tfor module in self.module_list:\n'
    ctx += '\t\t\tif isinstance(module, nn.AdaptiveAvgPool2d):\n'
    ctx += '\t\t\t\tx = module(_x)\n'
    ctx += '\t\t\t\tx = x.view(-1, 16)\n'
    ctx += '\t\t\telif isinstance(module, nn.Linear):\n'
    ctx += '\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\treturn x\n'
    ctx += '\t\t\telse:\n'
    ctx += '\t\t\t\tif first and not bn:\n'
    ctx += '\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\telif first and bn:\n'
    ctx += '\t\t\t\t\ŧx = module(x)\n'
    ctx += '\t\t\t\t\ŧ_x = self.relu(x)\n'
    ctx += '\t\t\t\t\tfirst = False\n'
    ctx += '\t\t\t\t\tbn = False\n'
    ctx += '\t\t\t\telse:\n'
    ctx += '\t\t\t\t\tif not odd and not bn:\n'
    ctx += '\t\t\t\t\t\tx = module(_x)\n'
    ctx += '\t\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\t\telif not odd and bn:\n'
    ctx += '\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\tx = self.relu(x)\n'
    ctx += '\t\t\t\t\t\todd = True\n'
    ctx += '\t\t\t\t\t\tbn = False\n'
    ctx += '\t\t\t\t\telse:\n'
    ctx += '\t\t\t\t\t\tif not bn:\n'
    ctx += '\t\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\t\t\telif bn:\n'
    ctx += '\t\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\t\t_x = _x + x\n'
    ctx += '\t\t\t\t\t\t\t_x = self.relu(_x)\n'
    ctx += '\t\t\t\t\t\t\todd = False\n'
    ctx += '\t\t\t\t\t\t\tbn = False\n'

    # ResNet50 definition
    ctx += 'def N2N(**kwargs):\n'
    ctx += '\tmodel = N2N(**kwargs)\n'
    ctx += '\treturn model\n'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("[INFO] Generating a new dense architecture...")
    f_out = open(os.path.join(out_dir, 'resnet_flat.py'), 'w')
    f_out.write(ctx)

    newcode = compile(ctx, "", 'exec')
    eval(newcode)