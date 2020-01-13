from src.custom_arch import layerUtil


def _genDenseArchResNet(model, out_dir, dense_chs, chs_map):
    ctx = 'import torch.nn as nn\n'
    ctx += 'import math'
    ctx += '__all__ = [\'resnet\']\n'
    ctx += 'class ResNet(nn.Module):\n'
    ctx += '\tdef __init__(self, num_classes=10):\n'
    ctx += '\t\tsuper(ResNet, self).__init__()\n'

    lyr = layerUtil(model, dense_chs)

    ctx += 'self.module_list = nn.ModuleList()'

    for module in model.module_list:
        ctx += lyr.getLayerDef(module)

