
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import numpy as np

from torch.autograd import Variable
from torch_geometric.nn import GCNConv


class Net(nn.Module):
    def __init__(self, kernel_size, num_filter):
        super(Net, self).__init__()

        self.kernel_size = 3
        self.num_filter = 48

        self.build()

    def build(self):
        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, 16)
        self.conv = nn.Conv1d(16, self.num_filter, self.kernel_size)

    def forward(self, traj, config):
        lngs = torch.unsqueeze(traj['lngs'], dim = 2)
        lats = torch.unsqueeze(traj['lats'], dim = 2)

        states = self.state_em(traj['states'].long())

        locs = torch.cat((lngs, lats, states), dim = 2)

        # map the coords into 16-dim vector
        locs = F.tanh(self.process_coords(locs))
        locs = locs.permute(0, 2, 1)
        #print('locs',locs.size())

        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # calculate the dist for local paths
        local_dist = utils.get_local_seq(traj['dist_gap'], self.kernel_size, config['dist_gap_mean'], config['dist_gap_std'])
        local_dist = torch.unsqueeze(local_dist, dim = 2)

        conv_locs = torch.cat((conv_locs, local_dist), dim = 2)

        return conv_locs
    
#%%

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        
        
    def forward(self, x, adj):
        #print(x.size())
        #print(adj.size())
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    


#%%
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #5 is the batch size
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, locs,adj):
        support = torch.matmul(locs, self.weight)
        
        #print('adj',adj.size())
        #print('su', support.size())
        output = torch.matmul(adj, support)
        
        #support = torch.mm(locs, self.weight)
        #output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'



    
#model = GCN(nfeat=features.shape[1],
 #           nhid=args.hidden,
  #          nclass=labels.max().item() + 1,
   #         dropout=args.dropout)

#output = model(features, adj)
    #%%
    

#print(torch.__version__)
#print(torch.version.cuda)
#torch.cuda.is_available()

#pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
#pip install torch-geometric