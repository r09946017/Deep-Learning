import torch
import torch.nn as nn

class Chomp1d(nn.Module):

	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
	
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation_size, dropout = 0.2):
		super(TemporalBlock, self).__init__()

		block = [
			Conv1d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation_size),
			Chomp1d(padding),
			nn.ReLU(replace = True),
			nn.Dropout(dropout),
			Conv1d(out_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation_size),
                        Chomp1d(padding),
                        nn.ReLU(replace = True),
                        nn.Dropout(dropout),
		]

		self.block = nn.Sequential(*block)

		self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
		self.relu = nn.ReLU(replace = True)

	def forward(self, x):
		out = self.block(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)


class TCN(nn.Module):

	def __init__(self, in_channels_list, out_channels_list, kernel_size = 2, dropout = 0.2):
		super(TCN, self).__init__()

		blocks = []

		for i in range(len(in_channels_list)):
			dilation_size = 1 << (i + 1) 

			blocks += [TemporalBlock(in_channels_list[i], out_channels_list[i], kernel_size, stride = 1, padding = (kernel_size - 1) * dilation_size, dilation_size = dilation_size, dropout = dropout)]

		self.blocks = nn.Sequential(*blocks)

	def forward(self, x):
		return self.blocks(x)
