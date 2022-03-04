import torch
import torch.nn as nn

OPS = {
	'none': lambda current_channels, stride, affine: Zero(stride),
	'avg_pool_3x3': lambda current_channels, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1,
	                                                                      count_include_pad=False),
	'max_pool_3x3': lambda current_channels, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
	'skip_connect': lambda current_channels, stride, affine: Identity() if stride == 1 else FactorizedReduce(
		current_channels, current_channels, affine=affine),
	'sep_conv_3x3': lambda current_channels, stride, affine: SepConv(current_channels, current_channels, 3, stride, 1,
	                                                                 affine=affine),
	'sep_conv_5x5': lambda current_channels, stride, affine: SepConv(current_channels, current_channels, 5, stride, 2,
	                                                                 affine=affine),
	'sep_conv_7x7': lambda current_channels, stride, affine: SepConv(current_channels, current_channels, 7, stride, 3,
	                                                                 affine=affine),
	'dil_conv_3x3': lambda current_channels, stride, affine: DilConv(current_channels, current_channels, 3, stride, 2,
	                                                                 2, affine=affine),
	'dil_conv_5x5': lambda current_channels, stride, affine: DilConv(current_channels, current_channels, 5, stride, 4,
	                                                                 2, affine=affine),
	'conv_7x1_1x7': lambda current_channels, stride, affine: nn.Sequential(
		nn.ReLU(inplace=False),
		nn.Conv2d(current_channels, current_channels, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
		nn.Conv2d(current_channels, current_channels, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
		nn.BatchNorm2d(current_channels, affine=affine)
	),
}


class ReLUConvBN(nn.Module):
	"""
		relu -> conv -> bn 操作 处理正常细胞的输入和缩小细胞的第二个输入
	"""

	def __init__(self, inputChannels, outputChannels, kernelSize, stride, padding, affine=True):
		super(ReLUConvBN, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(inputChannels, outputChannels, kernelSize, stride=stride, padding=padding, bias=False),
			nn.BatchNorm2d(outputChannels, affine=affine)
		)

	def forward(self, x):
		return self.op(x)


class DilConv(nn.Module):
	"""
		扩张卷积
	"""

	def __init__(self, inputChannels, outputChannels, kernelSize, stride, padding, dilation, affine=True):
		super(DilConv, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(inputChannels, inputChannels, kernel_size=kernelSize, stride=stride, padding=padding,
			          dilation=dilation, groups=inputChannels, bias=False),
			nn.Conv2d(inputChannels, outputChannels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(outputChannels, affine=affine),
		)

	def forward(self, x):
		return self.op(x)


class SepConv(nn.Module):
	"""
		分离卷积
	"""

	def __init__(self, inputChannels, outputChannels, kernelSize, stride, padding, affine=True):
		super(SepConv, self).__init__()
		self.op = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Conv2d(inputChannels, inputChannels, kernel_size=kernelSize, stride=stride, padding=padding,
			          groups=inputChannels, bias=False),
			nn.Conv2d(inputChannels, inputChannels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(inputChannels, affine=affine),
			nn.ReLU(inplace=False),
			nn.Conv2d(inputChannels, inputChannels, kernel_size=kernelSize, stride=1, padding=padding,
			          groups=inputChannels, bias=False),
			nn.Conv2d(inputChannels, outputChannels, kernel_size=1, padding=0, bias=False),
			nn.BatchNorm2d(outputChannels, affine=affine),
		)

	def forward(self, x):
		return self.op(x)


class Identity(nn.Module):
	"""
		直接连接 用于实现跳跃连接
	"""

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class Zero(nn.Module):
	"""
		0连接
	"""

	def __init__(self, stride):
		super(Zero, self).__init__()
		self.stride = stride

	def forward(self, x):
		return x.mul(0.) if self.stride == 1 else x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):
	"""
		处理缩小细胞的第一个输入
		把输出宽深度变为原来的一半
	"""

	def __init__(self, inputChannels, outputChannels, affine=True):
		super(FactorizedReduce, self).__init__()
		assert outputChannels % 2 == 0
		self.relu = nn.ReLU(inplace=False)
		# 这里可以解释为什么Net中设置缩小细胞要扩大一倍深度 因为是两个原始深度的拼接
		self.conv_1 = nn.Conv2d(inputChannels, outputChannels // 2, kernel_size=1, stride=2, padding=0, bias=False)
		self.conv_2 = nn.Conv2d(inputChannels, outputChannels // 2, kernel_size=1, stride=2, padding=0, bias=False)
		self.bn = nn.BatchNorm2d(outputChannels, affine=affine)

	def forward(self, x):
		x = self.relu(x)
		return self.bn(torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1))
