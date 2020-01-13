from .arch_utils import *

from .custom_resnet import _genDenseArchResnet

custom_arch = {
    'resnet':_genDenseArchResnet
}
