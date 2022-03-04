import torch
import scipy.io as scio
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


relu_test_x = scio.loadmat("relu_test_x.mat")
relu_test_y = scio.loadmat("relu_test_y.mat")

relu_test_input = torch.tensor(relu_test_x['x'], dtype = torch.float).cuda()
relu_test_target = torch.tensor(relu_test_y['y'], dtype = torch.float).cuda()
relu_test_input = relu_test_input[0:672]
relu_test_target = relu_test_target[0:672]

writer = SummaryWriter('relu_test/')

class ReluTestNet(nn.Module):
	def __init__(self):
		super(ReluTestNet, self).__init__()
		self.flow = nn.Sequential(
			nn.Linear(9, 2),
			nn.LeakyReLU(inplace=True),
		)
		self.out_flow = nn.Sequential(
			nn.Linear(2, 1),
			# nn.LeakyReLU(inplace=True),
		)

	def forward(self, x, fake_relu_para, writer=None,map=0):
		x = self.flow(x)
		data_map = torch.mm(x, fake_relu_para)
		flow_out = self.out_flow(data_map)
		if writer is not None:
			if map == 0:
				writer.add_image('ac_data_map_0', data_map[:, 0].reshape(1, 28, 24))
				writer.add_image('ac_data_map_1', data_map[:, 1].reshape(1, 28, 24))
			else:
				writer.add_image('death_data_map_0', data_map[:, 0].reshape(1, 28, 24))
				writer.add_image('death_data_map_1', data_map[:, 1].reshape(1, 28, 24))
		return flow_out

relu_test_net = ReluTestNet().cuda()

optimizer = torch.optim.SGD(
	relu_test_net.parameters(),
	lr=0.01
)
loss_f = nn.MSELoss().cuda()

for generate in range(0, 3000):
	optimizer.zero_grad()
	fake_relu_para = torch.tensor(([0.5, 0], [0, 0.5]), requires_grad=False).cuda()
	if generate == 1500:
		_out = relu_test_net(relu_test_input, fake_relu_para, writer)
	elif generate == 2000:
		fake_relu_para = torch.tensor(([1.0, 0], [0, 0]), requires_grad=False).cuda()
		_out = relu_test_net(relu_test_input, fake_relu_para, writer,map=1)
	_out = relu_test_net(relu_test_input, fake_relu_para)
	loss = loss_f(_out, relu_test_target)
	loss.backward()
	optimizer.step()
	writer.add_scalar('loss', loss.item(), generate)
