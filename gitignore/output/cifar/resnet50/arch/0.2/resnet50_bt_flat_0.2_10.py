import torch.nn as nn
import torch
__all__ = ['resnet50_bt_flat']
class ResNet50BT(nn.Module):
	def __init__(self, num_classes=10):
		super(ResNet50BT, self).__init__()
		self.conv1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(9)
		self.conv2 = nn.Conv2d(9, 5, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn2 = nn.BatchNorm2d(5)
		self.conv3 = nn.Conv2d(5, 6, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(6)
		self.conv4 = nn.Conv2d(6, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv5 = nn.Conv2d(9, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn5 = nn.BatchNorm2d(64)
		self.conv9 = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn9 = nn.BatchNorm2d(8)
		self.conv10 = nn.Conv2d(8, 10, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn10 = nn.BatchNorm2d(10)
		self.conv11 = nn.Conv2d(10, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn11 = nn.BatchNorm2d(64)
		self.conv12 = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn12 = nn.BatchNorm2d(2)
		self.conv13 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn13 = nn.BatchNorm2d(2)
		self.conv14 = nn.Conv2d(2, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn14 = nn.BatchNorm2d(64)
		self.conv15 = nn.Conv2d(64, 6, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn15 = nn.BatchNorm2d(6)
		self.conv16 = nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn16 = nn.BatchNorm2d(3)
		self.conv17 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn17 = nn.BatchNorm2d(64)
		self.conv18 = nn.Conv2d(64, 8, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn18 = nn.BatchNorm2d(8)
		self.conv19 = nn.Conv2d(8, 7, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn19 = nn.BatchNorm2d(7)
		self.conv20 = nn.Conv2d(7, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn20 = nn.BatchNorm2d(64)
		self.conv21 = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn21 = nn.BatchNorm2d(9)
		self.conv22 = nn.Conv2d(9, 12, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn22 = nn.BatchNorm2d(12)
		self.conv23 = nn.Conv2d(12, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn23 = nn.BatchNorm2d(64)
		self.conv24 = nn.Conv2d(64, 11, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn24 = nn.BatchNorm2d(11)
		self.conv25 = nn.Conv2d(11, 15, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn25 = nn.BatchNorm2d(15)
		self.conv26 = nn.Conv2d(15, 64, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn26 = nn.BatchNorm2d(64)
		self.conv27 = nn.Conv2d(64, 28, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn27 = nn.BatchNorm2d(28)
		self.conv28 = nn.Conv2d(28, 32, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn28 = nn.BatchNorm2d(32)
		self.conv29 = nn.Conv2d(32, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn29 = nn.BatchNorm2d(126)
		self.conv30 = nn.Conv2d(64, 126, kernel_size=1, stride=2, padding=0, bias=False)
		self.bn30 = nn.BatchNorm2d(126)
		self.conv31 = nn.Conv2d(126, 30, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn31 = nn.BatchNorm2d(30)
		self.conv32 = nn.Conv2d(30, 31, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn32 = nn.BatchNorm2d(31)
		self.conv33 = nn.Conv2d(31, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn33 = nn.BatchNorm2d(126)
		self.conv34 = nn.Conv2d(126, 30, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn34 = nn.BatchNorm2d(30)
		self.conv35 = nn.Conv2d(30, 31, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn35 = nn.BatchNorm2d(31)
		self.conv36 = nn.Conv2d(31, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn36 = nn.BatchNorm2d(126)
		self.conv37 = nn.Conv2d(126, 14, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn37 = nn.BatchNorm2d(14)
		self.conv38 = nn.Conv2d(14, 22, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn38 = nn.BatchNorm2d(22)
		self.conv39 = nn.Conv2d(22, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn39 = nn.BatchNorm2d(126)
		self.conv40 = nn.Conv2d(126, 17, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn40 = nn.BatchNorm2d(17)
		self.conv41 = nn.Conv2d(17, 9, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn41 = nn.BatchNorm2d(9)
		self.conv42 = nn.Conv2d(9, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn42 = nn.BatchNorm2d(126)
		self.conv43 = nn.Conv2d(126, 20, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn43 = nn.BatchNorm2d(20)
		self.conv44 = nn.Conv2d(20, 21, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn44 = nn.BatchNorm2d(21)
		self.conv45 = nn.Conv2d(21, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn45 = nn.BatchNorm2d(126)
		self.conv46 = nn.Conv2d(126, 31, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn46 = nn.BatchNorm2d(31)
		self.conv47 = nn.Conv2d(31, 17, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn47 = nn.BatchNorm2d(17)
		self.conv48 = nn.Conv2d(17, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn48 = nn.BatchNorm2d(126)
		self.conv49 = nn.Conv2d(126, 28, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn49 = nn.BatchNorm2d(28)
		self.conv50 = nn.Conv2d(28, 19, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn50 = nn.BatchNorm2d(19)
		self.conv51 = nn.Conv2d(19, 126, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn51 = nn.BatchNorm2d(126)
		self.conv52 = nn.Conv2d(126, 62, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn52 = nn.BatchNorm2d(62)
		self.conv53 = nn.Conv2d(62, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn53 = nn.BatchNorm2d(64)
		self.conv54 = nn.Conv2d(64, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn54 = nn.BatchNorm2d(140)
		self.conv55 = nn.Conv2d(126, 140, kernel_size=1, stride=2, padding=0, bias=False)
		self.bn55 = nn.BatchNorm2d(140)
		self.conv56 = nn.Conv2d(140, 55, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn56 = nn.BatchNorm2d(55)
		self.conv57 = nn.Conv2d(55, 62, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn57 = nn.BatchNorm2d(62)
		self.conv58 = nn.Conv2d(62, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn58 = nn.BatchNorm2d(140)
		self.conv59 = nn.Conv2d(140, 46, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn59 = nn.BatchNorm2d(46)
		self.conv60 = nn.Conv2d(46, 33, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn60 = nn.BatchNorm2d(33)
		self.conv61 = nn.Conv2d(33, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn61 = nn.BatchNorm2d(140)
		self.conv62 = nn.Conv2d(140, 60, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn62 = nn.BatchNorm2d(60)
		self.conv63 = nn.Conv2d(60, 18, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn63 = nn.BatchNorm2d(18)
		self.conv64 = nn.Conv2d(18, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn64 = nn.BatchNorm2d(140)
		self.conv65 = nn.Conv2d(140, 58, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn65 = nn.BatchNorm2d(58)
		self.conv66 = nn.Conv2d(58, 31, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn66 = nn.BatchNorm2d(31)
		self.conv67 = nn.Conv2d(31, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn67 = nn.BatchNorm2d(140)
		self.conv68 = nn.Conv2d(140, 51, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn68 = nn.BatchNorm2d(51)
		self.conv69 = nn.Conv2d(51, 25, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn69 = nn.BatchNorm2d(25)
		self.conv70 = nn.Conv2d(25, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn70 = nn.BatchNorm2d(140)
		self.conv71 = nn.Conv2d(140, 31, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn71 = nn.BatchNorm2d(31)
		self.conv72 = nn.Conv2d(31, 12, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn72 = nn.BatchNorm2d(12)
		self.conv73 = nn.Conv2d(12, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn73 = nn.BatchNorm2d(140)
		self.conv74 = nn.Conv2d(140, 58, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn74 = nn.BatchNorm2d(58)
		self.conv75 = nn.Conv2d(58, 50, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn75 = nn.BatchNorm2d(50)
		self.conv76 = nn.Conv2d(50, 140, kernel_size=1, stride=1, padding=0, bias=False)
		self.bn76 = nn.BatchNorm2d(140)
		self.avgpool = nn.AvgPool2d(8)
		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(140, num_classes)
	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		_x = self.relu(x)
		x = self.conv2(_x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		x = self.conv4(x)
		x = self.bn4(x)
		_x = self.conv5(_x)
		_x = self.bn5(_x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv9(_x)
		x = self.bn9(x)
		x = self.relu(x)
		x = self.conv10(x)
		x = self.bn10(x)
		x = self.relu(x)
		x = self.conv11(x)
		x = self.bn11(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv12(_x)
		x = self.bn12(x)
		x = self.relu(x)
		x = self.conv13(x)
		x = self.bn13(x)
		x = self.relu(x)
		x = self.conv14(x)
		x = self.bn14(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv15(_x)
		x = self.bn15(x)
		x = self.relu(x)
		x = self.conv16(x)
		x = self.bn16(x)
		x = self.relu(x)
		x = self.conv17(x)
		x = self.bn17(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv18(_x)
		x = self.bn18(x)
		x = self.relu(x)
		x = self.conv19(x)
		x = self.bn19(x)
		x = self.relu(x)
		x = self.conv20(x)
		x = self.bn20(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv21(_x)
		x = self.bn21(x)
		x = self.relu(x)
		x = self.conv22(x)
		x = self.bn22(x)
		x = self.relu(x)
		x = self.conv23(x)
		x = self.bn23(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv24(_x)
		x = self.bn24(x)
		x = self.relu(x)
		x = self.conv25(x)
		x = self.bn25(x)
		x = self.relu(x)
		x = self.conv26(x)
		x = self.bn26(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv27(_x)
		x = self.bn27(x)
		x = self.relu(x)
		x = self.conv28(x)
		x = self.bn28(x)
		x = self.relu(x)
		x = self.conv29(x)
		x = self.bn29(x)
		_x = self.conv30(_x)
		_x = self.bn30(_x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv31(_x)
		x = self.bn31(x)
		x = self.relu(x)
		x = self.conv32(x)
		x = self.bn32(x)
		x = self.relu(x)
		x = self.conv33(x)
		x = self.bn33(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv34(_x)
		x = self.bn34(x)
		x = self.relu(x)
		x = self.conv35(x)
		x = self.bn35(x)
		x = self.relu(x)
		x = self.conv36(x)
		x = self.bn36(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv37(_x)
		x = self.bn37(x)
		x = self.relu(x)
		x = self.conv38(x)
		x = self.bn38(x)
		x = self.relu(x)
		x = self.conv39(x)
		x = self.bn39(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv40(_x)
		x = self.bn40(x)
		x = self.relu(x)
		x = self.conv41(x)
		x = self.bn41(x)
		x = self.relu(x)
		x = self.conv42(x)
		x = self.bn42(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv43(_x)
		x = self.bn43(x)
		x = self.relu(x)
		x = self.conv44(x)
		x = self.bn44(x)
		x = self.relu(x)
		x = self.conv45(x)
		x = self.bn45(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv46(_x)
		x = self.bn46(x)
		x = self.relu(x)
		x = self.conv47(x)
		x = self.bn47(x)
		x = self.relu(x)
		x = self.conv48(x)
		x = self.bn48(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv49(_x)
		x = self.bn49(x)
		x = self.relu(x)
		x = self.conv50(x)
		x = self.bn50(x)
		x = self.relu(x)
		x = self.conv51(x)
		x = self.bn51(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv52(_x)
		x = self.bn52(x)
		x = self.relu(x)
		x = self.conv53(x)
		x = self.bn53(x)
		x = self.relu(x)
		x = self.conv54(x)
		x = self.bn54(x)
		_x = self.conv55(_x)
		_x = self.bn55(_x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv56(_x)
		x = self.bn56(x)
		x = self.relu(x)
		x = self.conv57(x)
		x = self.bn57(x)
		x = self.relu(x)
		x = self.conv58(x)
		x = self.bn58(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv59(_x)
		x = self.bn59(x)
		x = self.relu(x)
		x = self.conv60(x)
		x = self.bn60(x)
		x = self.relu(x)
		x = self.conv61(x)
		x = self.bn61(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv62(_x)
		x = self.bn62(x)
		x = self.relu(x)
		x = self.conv63(x)
		x = self.bn63(x)
		x = self.relu(x)
		x = self.conv64(x)
		x = self.bn64(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv65(_x)
		x = self.bn65(x)
		x = self.relu(x)
		x = self.conv66(x)
		x = self.bn66(x)
		x = self.relu(x)
		x = self.conv67(x)
		x = self.bn67(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv68(_x)
		x = self.bn68(x)
		x = self.relu(x)
		x = self.conv69(x)
		x = self.bn69(x)
		x = self.relu(x)
		x = self.conv70(x)
		x = self.bn70(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv71(_x)
		x = self.bn71(x)
		x = self.relu(x)
		x = self.conv72(x)
		x = self.bn72(x)
		x = self.relu(x)
		x = self.conv73(x)
		x = self.bn73(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.conv74(_x)
		x = self.bn74(x)
		x = self.relu(x)
		x = self.conv75(x)
		x = self.bn75(x)
		x = self.relu(x)
		x = self.conv76(x)
		x = self.bn76(x)
		_x = _x + x
		_x = self.relu(_x)
		x = self.avgpool(_x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x
def resnet50_bt_flat(**kwargs):
	model = ResNet50BT(**kwargs)
	return model
