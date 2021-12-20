# -*- coding: utf-8 -*-
"""
    Loss module for the project
    This defines a class for the loss to be calculated
    whilst training
"""

from libs import *
from constants import *


class MarginLoss(torch.nn.Module):
	"""
		This class defines the loss to be calculated
		during the training process
	"""

	def __init__(self, margin=1.0):
		"""
			Initial constructor class for the class
			Args:
				margin (float, optional): margin vale for the loss.
										  Defaults to 1.0.
		"""
		super(MarginLoss, self).__init__()
		self.margin = margin
		self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

	def forward(self, x, x_pos, x_neg):
		"""	
			Forward pass for the loss module
		Args:
			x (list): data
			x_pos (list): postive samples data
			x_neg (list): negtaive samples data

		Returns:
			(float): mean loss of all the samples
		"""
		fb1  = self.cos(x, x_pos)
		fb2  = self.cos(x, x_neg)
		loss = self.margin - fb1 + fb2
		loss = F.relu(loss)
		return loss.mean()
