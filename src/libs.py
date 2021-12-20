# -*- coding: utf-8 -*-
"""
    Libs Module for the project
    This module holds all the libariries required
    by other peices of the project
"""


import argparse
import pickle
import json, os, random, re, sys
from collections import defaultdict

import nltk
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from torch.autograd import Variable
from cnn import CNN_Text
from matplotlib import pyplot as plt



