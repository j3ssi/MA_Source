import os

from src.custom_arch import layerUtil


def _genDenseArchResNet(model, out_dir, dense_chs, chs_map, num_classes):
    print('\n\nDrin!!!!')
    ctx = 'import torch.nn as nn\n'
    ctx += 'import math'
    ctx += '__all__ = [\'resnet\']\n'
    ctx += 'class ResNet(nn.Module):\n'
    ctx += '\tdef __init__(self, num_classes=10):\n'
    ctx += '\t\tsuper(ResNet, self).__init__()\n'

    lyr = layerUtil(model, dense_chs, num_classes)

    ctx += '\t\tself.module_list = nn.ModuleList()'

    for module in model.module_list:
        ctx += lyr.getModuleDef(module)
        ctx += '\t\tmodule_list.append(layer)'

    ctx +='\tdef forward(self,x):'
    ctx +='\t\todd = False'
    ctx +='\t\tfirst = True'
    ctx +='\t\tbn = False'
    ctx +='\t\t_x = None'
    ctx +='\t\tprintNet = False'
    ctx +='\t\ti=0'
    ctx +='\t\tfor module in self.module_list:'
    ctx +='\t\t\tif isinstance(module, nn.AdaptiveAvgPool2d):'
    ctx +='\t\t\t\ttry:'
    ctx +='\t\t\t\t\tx = module(_x)'
    ctx +='\t\t\t\t\tx = x.view(-1, 16)'
    ctx +='\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\tprint("\navgpool", i)'
    ctx +='\t\t\t\t\t\ti = i + 1'
    ctx +='\t\t\t\texcept RuntimeError:'
    ctx +='\t\t\t\t\tprint("\n \n Oops!!!: ")'
    ctx +='\t\t\t\t\tprint("AvgPool")'
    ctx +='\t\t\telif isinstance(module, nn.Linear):'
    ctx +='\t\t\t\tx = module(x)'
    ctx +='\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\tprint("\nfc", i)'
    ctx +='\t\t\t\treturn x'
    ctx +='\t\t\telse:'
    ctx +='\t\t\t\tif first and not bn:'
    ctx +='\t\t\t\t\tx = module(x)'
    ctx +='\t\t\t\t\tbn = True'
    ctx +='\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\tprint("\nFirst conv", i)'
    ctx +='\t\t\t\t\t\ti = i+1'
    ctx +='\t\t\t\telif first and bn:'
    ctx +='\t\t\t\t\ŧx = module(x)'
    ctx +='\t\t\t\t\ŧ_x = self.relu(x)'
    ctx +='\t\t\t\t\ŧif printNet:'
    ctx +='\t\t\t\t\t\tprint("\nFirst bn", i)'
    ctx +='\t\t\t\t\t\ti = i+1'
    ctx +='\t\t\t\t\tfirst = False'
    ctx +='\t\t\t\t\tbn = False'
    ctx +='\t\t\t\telse:'
    ctx +='\t\t\t\t\tif not odd and not bn:'
    ctx +='\t\t\t\t\t\tx = module(_x)'
    ctx +='\t\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\t\tprint("\nconv",i)'
    ctx +='\t\t\t\t\t\t\ti=i+1'
    ctx +='\t\t\t\t\t\tbn = True'
    ctx +='\t\t\t\t\telif not odd and bn:'
    ctx +='\t\t\t\t\t\tx = module(x)'
    ctx +='\t\t\t\t\t\tx = self.relu(x)'
    ctx +='\t\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\t\tprint("\nbn",i)'
    ctx +='\t\t\t\t\t\t\ti=i+1'
    ctx +='\t\t\t\t\t\todd = True'
    ctx +='\t\t\t\t\t\tbn = False'
    ctx +='\t\t\t\t\telse:'
    ctx +='\t\t\t\t\t\tif not bn:'
    ctx +='\t\t\t\t\t\t\tx = module(x)'
    ctx +='\t\t\t\t\t\t\tbn = True'
    ctx +='\t\t\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\t\t\tprint("Odd conv",i)'
    ctx +='\t\t\t\t\t\t\t\ti=i+1'
    ctx +='\t\t\t\t\t\telif bn:'
    ctx +='\t\t\t\t\t\t\tx = module(x)'
    ctx +='\t\t\t\t\t\t\t_x = _x + x'
    ctx +='\t\t\t\t\t\t\t_x = self.relu(_x)'
    ctx +='\t\t\t\t\t\t\todd = False'
    ctx +='\t\t\t\t\t\t\tbn = False'
    ctx +='\t\t\t\t\t\t\tif printNet:'
    ctx +='\t\t\t\t\t\t\t\tprint("Odd bn",i)'
    ctx +='\t\t\t\t\t\t\t\ti=i+1'


  # ResNet50 definition
    ctx += 'def resnet50_flat(**kwargs):\n'
    ctx += '\tmodel = ResNet50(**kwargs)\n'
    ctx += '\treturn model\n'
    if not os.path.exists():
        os.makedirs(out_dir)

    print ("[INFO] Generating a new dense architecture...")
    f_out = open(os.path.join(out_dir, 'resnet_flat.py'),'w')
    f_out.write(ctx)
