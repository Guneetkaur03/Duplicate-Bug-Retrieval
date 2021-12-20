# -*- coding: utf-8 -*-
"""
	Helper Module for data for the project
	This holds all the misc methods that are used all over
	the project.
"""

from libs import *
from constants import *

def to_one_hot(idx, size):
	"""
		This method one-hot encoded the data

		Args:
			idx (str): id of the data
			size (int): size of the data

		Returns:
			(np.ndarray): one-hot encoded array
	"""
	one_hot = np.zeros(size)
	one_hot[int(float(idx))] = 1
	return one_hot


def read_bug_ids(data):
	"""
		This method read ids of the bugs
		Args:
			data (str): path of the data folder
		Returns:
			(list): bug_ids appended to the list
	"""
	bug_ids = []
	with open(os.path.join(data, 'bug_ids.txt'), 'r') as f:
		for line in f:
			bug_ids.append(int(line.strip()))
	return bug_ids


def get_neg_bug(invalid_bugs, bug_ids):
	"""
		This method returns a valid bug.
		Args:
			invalid_bugs (list): ids of list of all the invalid bugs
			bug_ids (list): ids of list of all the bugs

		Returns:
			(int): one random valid bug
	"""
	neg_bug = random.choice(bug_ids)
	while neg_bug in invalid_bugs:
		neg_bug = random.choice(bug_ids)
	return neg_bug


def data_padding(data, max_seq_length):
	"""
		This method pads the data to a length
		Args:
			data (list): list of arrays
			max_seq_length ([type]): [description]

		Returns:
			(np.ndarray): padded array
	"""
	seq_lengths = [len(seq) for seq in data]
	seq_lengths.append(6)
	max_seq_length = min(max(seq_lengths), max_seq_length)
	padded_data    = np.zeros(shape=[len(data), max_seq_length])
	for i, seq in enumerate(data):
		seq = seq[:max_seq_length]
		for j, token in enumerate(seq):
			padded_data[i, j] = int(token)
	return padded_data.astype(np.int)


def read_batch_bugs(batch_bugs, data, test=False):
	"""
		This method reades the bugs divided into batches, and
		returns the dictionary for the same.

		Args:
			batch_bugs (list): ids of the bugs
			data (str): path to the data folder
			test (bool, optional): To set the value of test. Defaults to False.

		Returns:
			(dict): cobined dataset
	"""
	desc_word = []
	desc_char = []
	short_desc_word = []
	short_desc_char = []
	info = []
	for bug_id in batch_bugs:
		bug = pickle.load(open(os.path.join(data, 'bugs', '{}.pkl'.format(bug_id)), 'rb'))
		desc_word.append(bug['description_word'])
		desc_char.append(bug['description_char'])
		short_desc_word.append(bug['short_desc_word'])
		short_desc_char.append(bug['short_desc_char'])
		info_ = np.concatenate((
			to_one_hot(bug['bug_severity'], info_dict['bug_severity']),
			to_one_hot(bug['bug_status'], info_dict['bug_status']),
			to_one_hot(bug['component'], info_dict['component']),
			to_one_hot(bug['priority'], info_dict['priority']),
			to_one_hot(bug['product'], info_dict['product']),
			to_one_hot(bug['version'], info_dict['version'])))
		info.append(info_)

	if cuda:
		desc_word = Variable(torch.from_numpy(data_padding(desc_word, 500)), volatile=test).cuda()
		desc_char = Variable(torch.from_numpy(data_padding(desc_char, 2000)), volatile=test).cuda()
		short_desc_word = Variable(torch.from_numpy(data_padding(short_desc_word, 100)), volatile=test).cuda()
		short_desc_char = Variable(torch.from_numpy(data_padding(short_desc_char, 400)), volatile=test).cuda()
		info = Variable(torch.from_numpy(np.array(info)), volatile=test).cuda()
	else:
		desc_word = Variable(torch.from_numpy(data_padding(desc_word, 500)), volatile=test)
		desc_char = Variable(torch.from_numpy(data_padding(desc_char, 2000)), volatile=test)
		short_desc_word = Variable(torch.from_numpy(data_padding(short_desc_word, 100)), volatile=test)
		short_desc_char = Variable(torch.from_numpy(data_padding(short_desc_char, 400)), volatile=test)
		info = Variable(torch.from_numpy(np.array(info)), volatile=test)

	batch_bugs = dict()
	batch_bugs['info'] = info
	batch_bugs['desc'] = (desc_word, desc_char)
	batch_bugs['short_desc'] = (short_desc_word, short_desc_char)

	return batch_bugs


def read_batch_triplets(batch_triplets, data):
	"""
		This module reads the batch data in triples

		Args:
			batch_triplets (list): data in triplets
			data (str): path to data

		Returns:
			(tuple): tuple of triple batches
	"""
	batch_input_bugs = []
	batch_pos_bugs = []
	batch_neg_bugs = []
	for triplet in batch_triplets:
		batch_input_bugs.append(triplet[0])
		batch_pos_bugs.append(triplet[1])
		batch_neg_bugs.append(triplet[2])
	return 	read_batch_bugs(batch_input_bugs, data), \
			read_batch_bugs(batch_pos_bugs, data), \
			read_batch_bugs(batch_neg_bugs, data)


def read_test_data(data):
	"""
		this method reads the test data

		Args:
			data (str): Path to the data folder

		Returns:
			(list): list containg the read data
			(set): set of unique ids
	"""
	test_data = []
	bug_ids = set()
	with open(os.path.join(data, 'test.txt'), 'r') as f:
		for line in f:
			tokens = line.strip().split()
			test_data.append([int(tokens[0]), [int(bug) for bug in tokens[1:]]])
			for token in tokens:
				bug_ids.add(int(token))
	return test_data, list(bug_ids)


def read_train_data(data):
	"""
		this method reads the train data

		Args:
			data (str): Path to the data folder

		Returns:
			(list): list containg the read data
			(dict): set of duplicate data
	"""
	data_pairs = []
	data_dup_sets = {}
	with open(os.path.join(data, 'train.txt'), 'r') as f:
		for line in f:
			bug1, bug2 = line.strip().split()
			data_pairs.append([int(bug1), int(bug2)])
			if int(bug1) not in data_dup_sets.keys():
				data_dup_sets[int(bug1)] = set()
			data_dup_sets[int(bug1)].add(int(bug2))
	return data_pairs, data_dup_sets





def batch_iterator(data, batch_size, n_neg):
	"""
		This yeilds an iteration from each batch

		Args:
			data (str): path of the data folder
			batch_size (int): the batch size
			n_neg (int): number negation bugs

		Yields:
			(tqdm): tqdm loop
			(tuple): triplets data
	"""
	global train_data
	global dup_sets

	if not train_data:
		train_data, dup_sets = read_train_data(data)

	random.shuffle(train_data)
	bug_ids = read_bug_ids(data)
	num_batches = int(len(train_data) / batch_size)

	if len(data) % batch_size > 0:
		num_batches += 1

	loop = tqdm(range(num_batches))
	loop.set_description('Training')
	for i in loop:
		batch_triplets = []
		for j in range(batch_size):
			offset = batch_size * i + j
		if offset >= len(train_data):
			break
		for i in range(n_neg):
			neg_bug = get_neg_bug(dup_sets[train_data[offset][0]], bug_ids)
			batch_triplets.append([train_data[offset][0], train_data[offset][1], neg_bug])
	yield loop, read_batch_triplets(batch_triplets, data)
