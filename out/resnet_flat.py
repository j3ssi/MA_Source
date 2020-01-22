import torch.nn as nn
import math
__all__ = ['N2N']
class N2N(nn.Module):
	def __init__(self, num_classes=10):
		super(N2N, self).__init__()
		self.module_list = nn.ModuleList()
		layer = nn.Conv2d(module.conv1.weight, 3, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn1.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv2.weight, 14, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn2.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv3.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn3.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv4.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn4.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv5.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn5.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv6.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn6.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv7.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn7.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv8.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn8.weight)
		module_list.append(layer)
		layer = nn.Conv2d(module.conv9.weight, 16, kernel_size=16, stride=(3, 3), padding=(1, 1), bias=(1, 1))
		module_list.append(layer)
		layer = nn.BatchNorm2d(module.bn9.weight)
		module_list.append(layer)
		layer = nn.AdaptiveAvgPool2d((1, 1))
		module_list.append(layer)
		layer = nn.Linear(16, 10)
		module_list.append(layer)
	def forward(self,x):
		odd = False
		first = True
		bn = False
		_x = None
		for module in self.module_list:
			if isinstance(module, nn.AdaptiveAvgPool2d):
				x = module(_x)
				x = x.view(-1, 16)
			elif isinstance(module, nn.Linear):
				x = module(x)
				return x
			else:
				if first and not bn:
					x = module(x)
					bn = True
				elif first and bn:
				\ŧx = module(x)
				\ŧ_x = self.relu(x)
					first = False
					bn = False
				else:
					if not odd and not bn:
						x = module(_x)
						bn = True
					elif not odd and bn:
						x = module(x)
						x = self.relu(x)
						odd = True
						bn = False
					else:
						if not bn:
							x = module(x)
							bn = True
						elif bn:
							x = module(x)
							_x = _x + x
							_x = self.relu(_x)
							odd = False
							bn = False
def N2N(**kwargs):
	model = N2N(**kwargs)
	return model
