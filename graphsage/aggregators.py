import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import random
import numpy as np

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack) 
        # Comment from George for ^^: Wait this optimization actually works? That's actually pretty cool.
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class MaxPoolAggregator(nn.Module):
    """
    Aggregates a node's embeddings using max pooling of the output of a trainable
    single layer neural network
    """
    def __init__(self, features, input_dim, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        NOTE: It is assumed that cuda is Flase and gcn is True since that's the parameters we're using.
        """

        super(MaxPoolAggregator, self).__init__()
        self.features = features
        self.cuda = cuda
        self.gcn = gcn

        self.W_pool = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        self.b = nn.Parameter(torch.FloatTensor(input_dim))
        nn.init.xavier_uniform_(self.W_pool)
        #nn.init.xavier_uniform_(self.b)

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        _sample = random.sample
        samp_neighs = [_sample(to_neigh, num_sample)
                       if len(to_neigh) >= num_sample
                       else list(to_neigh)
                       for to_neigh in to_neighs]

        mask = torch.zeros(len(samp_neighs), num_sample+1, 1)
        for i, samp_neigh in enumerate(samp_neighs):
            for j, neigh in enumerate(samp_neigh):
                mask[i, j, 0] = 1

        # for below, repeal until samp_neigh is length n_samples + 1 for the purpose of constant dimensionality
        samp_neighs = [samp_neigh + [nodes[i].item() for _ in range(num_sample - len(samp_neigh) + 1)]
                       for i, samp_neigh in enumerate(samp_neighs)]

        unique_nodes_list = list(set([n for neigh in samp_neighs for n in neigh]))
        unique_nodes = {n:i for i, n in enumerate(unique_nodes_list)}
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        for i, samp_neigh in enumerate(samp_neighs):
            for j, neigh in enumerate(samp_neigh):
                samp_neighs[i][j] = unique_nodes[neigh]

        neigh_embedded = torch.stack([embed_matrix[samp_neigh, :] for samp_neigh in samp_neighs])

        # dimensionality of neigh_embeded is batch_sz, num_sample+1, input_dum
        neigh_layer_out = F.leaky_relu(neigh_embedded @ self.W_pool + self.b)

        max_out, _ = torch.max(neigh_layer_out * mask, dim=1)

        return max_out
        

