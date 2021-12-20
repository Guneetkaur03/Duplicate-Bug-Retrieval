# -*- coding: utf-8 -*-
"""
	Baseline module for the poject.
	It defines the baseline network to be trained
	on the data
"""

from libs import *
from constants import *
from helper_functions import load_emb_matrix

class BaseNet(nn.Module):
	"""
		This class deinfes the baseline model for
		training. Inherited from nn.Module for neural networks by Torch
	"""
	def __init__(self):
		"""
			This is the construction intialization for the
			class
			Args:
				(None)
			Returns:
				(None)
		"""
		super(BaseNet, self).__init__()
		self.word_embed = nn.Embedding(n_words, word_dim, max_norm=1, padding_idx=0)
		self.word_embed.weight = nn.Parameter(
		torch.from_numpy(load_emb_matrix(n_words, word_dim, data)).float()
		)
		
		self.CNN = CNN_Text(word_dim, n_filters)
		self.RNN = nn.GRU(input_size=word_dim, hidden_size=50, bidirectional=True, batch_first=True)

		self.info_proj = nn.Sequential(nn.Linear(n_prop, 100), nn.Tanh())
		self.projection = nn.Linear(300, 100)


	def forward(self, x):
		"""
		 	Forward pass for the model
			Args:
				x (list): [info, desc, short desc] data
			Returns:
				(obj): embeddings object applied
		"""
		
		info = x['info']
		info_feature = self.info_proj(info.float())

		desc = x['desc'][0]
		desc_feature = self.CNN(self.word_embed(desc))

		short_desc = x['short_desc'][0]
		out, hidden = self.RNN(self.word_embed(short_desc))
		short_desc_feature = torch.mean(out, dim=1)
		
		feature = torch.cat([info_feature, short_desc_feature, desc_feature], -1)
		return self.projection(feature)
