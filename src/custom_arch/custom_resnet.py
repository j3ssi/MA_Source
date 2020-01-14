import os

from src.custom_arch import layerUtil


def _genDenseArchResNet(model, out_dir, dense_chs, chs_map, num_classes):
    ctx = 'import torch.nn as nn\n'
    ctx += 'import math'
    ctx += '__all__ = [\'resnet\']\n'
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
    j = 0
    for name, module in model.module_list.named_modules():
        print("\n Name: ", name)
        i = i+1
        # ctx += lyr.getModuleDef(module,param)
        # ctx += '\t\tmodule_list.append(layer)\n'

    ctx += '\tdef forward(self,x):\n'
    ctx += '\t\todd = False\n'
    ctx += '\t\tfirst = True\n'
    ctx += '\t\tbn = False\n'
    ctx += '\t\t_x = None\n'
    ctx += '\t\tprintNet = False\n'
    ctx += '\t\ti=0\n'
    ctx += '\t\tfor module in self.module_list:\n'
    ctx += '\t\t\tif isinstance(module, nn.AdaptiveAvgPool2d):\n'
    ctx += '\t\t\t\ttry:\n'
    ctx += '\t\t\t\t\tx = module(_x)\n'
    ctx += '\t\t\t\t\tx = x.view(-1, 16)\n'
    ctx += '\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\tprint("\navgpool", i)\n'
    ctx += '\t\t\t\t\t\ti = i + 1\n'
    ctx += '\t\t\t\texcept RuntimeError:\n'
    ctx += '\t\t\t\t\tprint("\n \n Oops!!!: ")\n'
    ctx += '\t\t\t\t\tprint("AvgPool")\n'
    ctx += '\t\t\telif isinstance(module, nn.Linear):\n'
    ctx += '\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\tprint("\nfc", i)\n'
    ctx += '\t\t\t\treturn x\n'
    ctx += '\t\t\telse:\n'
    ctx += '\t\t\t\tif first and not bn:\n'
    ctx += '\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\tprint("\nFirst conv", i)\n'
    ctx += '\t\t\t\t\t\ti = i+1\n'
    ctx += '\t\t\t\telif first and bn:\n'
    ctx += '\t\t\t\t\ŧx = module(x)\n'
    ctx += '\t\t\t\t\ŧ_x = self.relu(x)\n'
    ctx += '\t\t\t\t\ŧif printNet:\n'
    ctx += '\t\t\t\t\t\tprint("\nFirst bn", i)\n'
    ctx += '\t\t\t\t\t\ti = i+1\n'
    ctx += '\t\t\t\t\tfirst = False\n'
    ctx += '\t\t\t\t\tbn = False\n'
    ctx += '\t\t\t\telse:\n'
    ctx += '\t\t\t\t\tif not odd and not bn:\n'
    ctx += '\t\t\t\t\t\tx = module(_x)\n'
    ctx += '\t\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\t\tprint("\nconv",i)\n'
    ctx += '\t\t\t\t\t\t\ti=i+1\n'
    ctx += '\t\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\t\telif not odd and bn:\n'
    ctx += '\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\tx = self.relu(x)\n'
    ctx += '\t\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\t\tprint("\nbn",i)\n'
    ctx += '\t\t\t\t\t\t\ti=i+1\n'
    ctx += '\t\t\t\t\t\todd = True\n'
    ctx += '\t\t\t\t\t\tbn = False\n'
    ctx += '\t\t\t\t\telse:\n'
    ctx += '\t\t\t\t\t\tif not bn:\n'
    ctx += '\t\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\t\tbn = True\n'
    ctx += '\t\t\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\t\t\tprint("Odd conv",i)\n'
    ctx += '\t\t\t\t\t\t\t\ti=i+1\n'
    ctx += '\t\t\t\t\t\telif bn:\n'
    ctx += '\t\t\t\t\t\t\tx = module(x)\n'
    ctx += '\t\t\t\t\t\t\t_x = _x + x\n'
    ctx += '\t\t\t\t\t\t\t_x = self.relu(_x)\n'
    ctx += '\t\t\t\t\t\t\todd = False\n'
    ctx += '\t\t\t\t\t\t\tbn = False\n'
    ctx += '\t\t\t\t\t\t\tif printNet:\n'
    ctx += '\t\t\t\t\t\t\t\tprint("Odd bn",i)\n'
    ctx += '\t\t\t\t\t\t\t\ti=i+1\n'

    # ResNet50 definition
    ctx += 'def N2N(**kwargs):\n'
    ctx += '\tmodel = N2N(**kwargs)\n'
    ctx += '\treturn model\n'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("[INFO] Generating a new dense architecture...")
    f_out = open(os.path.join(out_dir, 'resnet_flat.py'), 'w')
    f_out.write(ctx)
