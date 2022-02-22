import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

	def __init__(self, in_features):
		super(ResidualBlock, self).__init__()

		block = [
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features),
			nn.ReLU(inplace = True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(in_features, in_features, 3),
			nn.InstanceNorm2d(in_features)
		]

		self.block = nn.Sequential(*block)

	def forward(self, x):
		return (x + self.block(x))

class Generator(nn.Module):

	def __init__(self):
		super(Generator, self).__init__()
 
		model = [
			nn.ReflectionPad2d(3),
			nn.Conv2d(3, 64, 7),
			nn.InstanceNorm2d(64),
			nn.ReLU(inplace = True)
		]

		### Downsampling
		in_features, out_features = 64, 128
		for _ in range(2):
			model += [
				nn.Conv2d(in_features, out_features, 3, stride = 2, padding = 1),
				nn.InstanceNorm2d(out_features),
				nn.ReLU(inplace = True)
			]
			in_features = out_features
			out_features *= 2
		
		### Residual Blocks
		for _ in range(9):
			model += [
				ResidualBlock(in_features)
			]

		### Upsampling
		out_features = in_features // 2
		for _ in range(2):
			model += [
				nn.ConvTranspose2d(in_features, out_features, 3, stride = 2, padding = 1, output_padding = 1),
				nn.InstanceNorm2d(out_features),
				nn.ReLU(inplace = True)
			]
			in_features = out_features,
			out_feautres //= 2

		model += [
			nn.ReflectionPad2d(3),
			nn.Conv2d(64, 3, 7),
			nn.Tanh()
		]

		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)

class Discriminator(nn.Module):

	def __init__(self):
		super(Discriminator, self).__init__()

		model = [
			nn.Conv2d(3, 64, 4, stride = 2, padding = 1),
			nn.LeakyReLU(0.2, inplace = True),
		]
		
		in_feautres, out_features = 64, 128
		for _ in range(3):
			model += [
				nn.Conv2d(in_features, out_features, 4, stride = 2, padding = 1),
				nn.InstanceNorm2d(out_features),
				nn.LeakyReLU(0.2, inplace = True)
			]
			in_features = out_features
			out_features *= 2
		
		model += [
			nn.Conv2d(in_features, 1, 4, padding = 1)
		]
		
		self.model = nn.Sequential(*model)

	def forward(self, x):
		return self.model(x)
