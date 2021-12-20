# -*- coding: utf-8 -*-
"""
	CNN Class to be used in the project.
	It defines the the basic CNN with all the basic configurations
"""
from libs import *
from constants import *

class CNN_Text(nn.Module):
	"""
		This class defines a CNN based on nn.Module class
		which is a base class for Torch
	"""
	def __init__(self, input_dim, n_filters):
		"""	
			Initalizing constructor method for the class
			Args:
				input_dim (int): Input dimensions of the layer
				n_filters (int): Number of filters of the layer
		"""
		super(CNN_Text, self).__init__()
		D = input_dim
		Ci = 1
		Co = n_filters
		Ks = [3, 4, 5]
		self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
		self.fc = nn.Sequential(nn.Linear(n_filters * 3, 100), nn.Tanh())

	def forward(self, x):
		"""
			Forward pass for the CNN layer
			Args:
				x (list): (N, Ci, W, D) data
			Returns:
				(obj): Object of the activation function
		"""
		x = x.unsqueeze(1)  # 
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # [(N, Co, W), ...]*len(Ks)
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] # [(N, Co), ...]*len(Ks)
		x = torch.cat(x, 1)
		return self.fc(x)