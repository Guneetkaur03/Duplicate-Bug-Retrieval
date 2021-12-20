# -*- coding: utf-8 -*-
"""
    Constants Module for the project
    This keeps the values of the all the constants used 
    all over in the project.
"""

from libs import *

train_data = None
dup_sets = None

data     = './data/'
vocab_f  = './data/eclipse/word_vocab.pkl'
glove_f  = './data/embedding/glove.42B.{}d.txt'
top_k    = 25
epochs   = 30
base_model = True
n_words  = 20000
n_chars  = 100
word_dim = 300
char_dim = 50
n_filters = 64
n_prop   = 1123
n_neg    = 1
batchsize = 64
learning_rate = 1e-3

cuda = torch.cuda.is_available()
feature_ext    = '_features.t7'
checkpoint_ext = '_checkpoint.t7'

info_dict = {
                'bug_severity': 7,
                'bug_status': 3,
                'component': 612,
                'priority': 5,
                'product': 170,
                'version': 326
            }
