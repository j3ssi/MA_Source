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

import os
from .arch_utils import layerUtil

arch = {}
for i in range(1, 14):
    conv_idx = (i-1)*2
    bn_idx = conv_idx +1
    arch[conv_idx] = {'name':'conv'+str(i) , 'kernel_size':3, 'stride':1, 'padding':1, 'bias':False}
    arch[bn_idx] = {'name':'bn'+str(i)}

arch[26] = {'name':'pool', 'kernel_size':2, 'stride':2}
arch[27] = {'name':'avgpool_adt', 'num':'(7, 7)'}
arch[28] = {'name':'relu'}
arch[29] = {'name':'fc1'}
arch[30] = {'name':'fc2'}
arch[31] = {'name':'fc3'}
arch[32] = {'name':'dropout'}

def _genDenseArchVGG16(model, out_f_dir1, out_f_dir2, arch_name, dense_chs, chs_map=None):

  # File heading
  ctx = 'import torch.nn as nn\n'
  ctx += '__all__ = [\'vgg16_flat\']\n'
  ctx += 'class VGG16(nn.Module):\n'
  ctx += '\tdef __init__(self, num_classes=1000):\n'
  ctx += '\t\tsuper(VGG16, self).__init__()\n'

  lyr = layerUtil(model, dense_chs)

  # Layer definition
  for idx in sorted(arch):
    ctx += lyr.getLayerDef(arch[idx])

  ctx += '\tdef forward(self, x):\n'
  ctx += lyr.forward('conv1')
  ctx += lyr.forward('bn1')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv2')
  ctx += lyr.forward('bn2')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv3')
  ctx += lyr.forward('bn3')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv4')
  ctx += lyr.forward('bn4')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv5')
  ctx += lyr.forward('bn5')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv6')
  ctx += lyr.forward('bn6')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv7')
  ctx += lyr.forward('bn7')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv8')
  ctx += lyr.forward('bn8')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv9')
  ctx += lyr.forward('bn9')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv10')
  ctx += lyr.forward('bn10')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('pool')

  ctx += lyr.forward('conv11')
  ctx += lyr.forward('bn11')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv12')
  ctx += lyr.forward('bn12')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('conv13')
  ctx += lyr.forward('bn13')
  ctx += lyr.forward('relu')
  ctx += lyr.forward('avgpool_adt')
  
  ctx += '\t\tx = x.view(x.size(0), -1)\n'
  ctx += lyr.forward('fc1')
  ctx += lyr.forward('relu')
  ctx += '\t\tx = self.dropout(x)\n'
  ctx += lyr.forward('fc2')
  ctx += lyr.forward('relu')
  ctx += '\t\tx = self.dropout(x)\n'
  ctx += lyr.forward('fc3')
  ctx += '\t\treturn x\n'

  # AlexNet definition
  ctx += 'def vgg16_flat(**kwargs):\n'
  ctx += '\tmodel = VGG16(**kwargs)\n'
  ctx += '\treturn model\n'

  if not os.path.exists(out_f_dir2):
      os.makedirs(out_f_dir2)

  print ("[INFO] Generating a new dense architecture...")
  f_out1 = open(os.path.join(out_f_dir1, 'vgg16_flat.py'),'w')
  f_out1.write(ctx)
  f_out2 = open(os.path.join(out_f_dir2, arch_name),'w')
  f_out2.write(ctx)


