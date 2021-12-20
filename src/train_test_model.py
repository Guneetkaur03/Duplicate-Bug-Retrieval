# -*- coding: utf-8 -*-
"""
    Train Test Module for the project
    This module holds moethod to train or test the model.
    It also divides the data into features and arrange them according 
    to the batches.
"""
import helper_data
import baseline
import proposed
from loss import *
from constants import *
from helper_data import *

def train_network(net, optimizer):
    """
        This module all training steps for the module.
        It takes all the reqd args and returns the mean loss.
        Args:
            net (nn.Module): Class object of torch model (base/proposed)
            optimizer (Adam): optimizer for the loss

        Returns:
            (float): mean of all the losses in thet epoch
    """
    net.train()
    losses = []
    margin = MarginLoss(margin=1.0)
    #loop over all batches in the epoch
    for loop, (batch_x, batch_pos, batch_neg) \
        in helper_data.batch_iterator(data, batchsize, n_neg):
        #calculate loss of all batch polarity
        loss = margin(net(batch_x), net(batch_pos), net(batch_neg))
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
        optimizer.zero_grad()
        #perform backprop step
        loss.backward()
        optimizer.step()

    return np.array(losses).mean()


def export(net, data):
    """
        This module reads the data file and exports the features

        Args:
            net (nn.Module): Class object of torch model (base/proposed)
            data (str): path to data folder

        Returns:
            (dict): features extracted into a dict
    """
    _, bug_ids = read_test_data(data)
    features = {}

    batch_size = batchsize * n_neg
    num_batch = int(len(bug_ids) / batch_size)
    if len(bug_ids) % batch_size > 0:
        num_batch += 1

    for i in tqdm(range(num_batch), desc="Exporting Features"):
        batch_ids = []
        for j in range(batch_size):
            offset = batch_size * i + j
            if offset >= len(bug_ids):
                break
            batch_ids.append(bug_ids[offset])
        batch_features = net(read_batch_bugs(batch_ids, data, test=True))
        for bug_id, feature in zip(batch_ids, batch_features):
            features[bug_id] = feature
    return features


def test_network(data, top_k, net_type, features = None):
    """
        This module tests the loaded features/or loads them and
        then run predicitions to calculate the recall score.
        Note, recall = true_positives/total_samples

        Args:
            data (str): path to data folder
            top_k (int): top k similar bugs
            net_type (str): type of the model
            features (dict, optional): extracted features. Defaults to None.

        Returns:
            (float): recall score 
    """
    #load features if not present
    if not features:
        features = torch.load(str(net_type) + feature_ext)
    
    #cosine similarity
    cosine_batch   = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    #keep a count of correct and total samples
    true_positives = 0
    total_samples  = 0
    #load samples
    samples_       = torch.stack(features.values())
    #load test data
    test_samples, _ = read_test_data(data)
    
    #loop around samples
    #TODO: write one line comment of each step
    for i in tqdm(range(len(test_samples)), desc = "Running Tests"):

        query_sample = test_samples[i][0]
        ground_truth = test_samples[i][1]

        query_ = features[query_sample].expand(samples_.size(0), samples_.size(1))
        cos_   = cosine_batch(query_, samples_)

        (_, indices) = torch.topk(cos_, k=top_k + 1)
        candidates   = [features.keys()[x.data[0]] for x in indices]

        true_positives += len(set(candidates) & set(ground_truth))
        total_samples  += len(ground_truth)

    return float(true_positives) / total_samples


def run_training(net_type):
    """
        This module runs training of th defined model

        Args:
            net_type (str): type of the model
        Returns:
            (None)
    """
    #check if the feautres exists
    # and test the model
    if os.path.exists(str(net_type) + feature_ext):
        print("Testing Model!")
        print('Final recall@{}={:.4f}'.format(top_k, test_network(data, top_k, net_type)))
        sys.exit(0)

    #check if the checkpoint exists
    if os.path.exists(str(net_type) + checkpoint_ext):
        trained_net = torch.load(str(net_type) + checkpoint_ext)
        torch.save(export(trained_net, data), str(net_type) + feature_ext)
        print('Final recall@{}={:.4f}'.format(top_k, test_network(data, top_k, net_type)))
        sys.exit(0)

    if net_type == 'baseline':
        net = baseline.BaseNet()
    else:
        net = proposed.Net()
    if cuda:
        net.cuda()
    optimizer   = optim.Adam(net.parameters(), lr = learning_rate)
    best_recall = 0
    best_epoch  = 0
    losses = []
    # create tqdm object for epochs
    epochs_tqdm = tqdm(range(1, epochs + 1), desc="Training Model")
    for epoch in epochs_tqdm:
        epochs_tqdm.set_description("Epoch {}".format(epoch))
        # optimize params on every 10th epoch
        if epoch == 10:
            optimizer = optim.Adam(net.parameters(), lr = learning_rate * 0.1)
        loss     = train_network(net, optimizer)
        losses.append(loss)
        features = export(net, data)
        recall   = test_network(data, top_k, net_type, features)
        print('Loss={:.4f}, Recall@{}={:.4f}'.format(loss, top_k, recall))
        if recall > best_recall:
            best_recall = recall
            best_epoch  = epoch
            torch.save(net, str(net_type) + checkpoint_ext)
            torch.save(features, str(net_type) + feature_ext)

    
    plt.plot(range(epochs), losses)
    plt.xlabel('Epcohs')
    plt.ylabel('loss')
    plt.title('Loss v/s Epoch Curve')
    plt.show()
    print('Best_epoch={}, Best_recall={:.4f}'.format(best_epoch, best_recall))