import pickle as pkl

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import MessagePassing
import argparse
import warnings
class AggNet(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)
        # self.parameter = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1), ))
        self.parameter2 = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1), ))
        self.parameter3 = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1,1),))
        # self.parameter4 = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(1)).reshape(1, 1), ))

    def forward(self, x, edge_index, edge_weight):
        out = self.propagate(edge_index=edge_index, edge_weight=edge_weight, x=x)

        return F.tanh(out)

    def message(self, x_i, x_j, edge_weight):
        return (abs(self.parameter2) * x_j + abs(self.parameter3) * x_i)


# F.relu(self.parameter)*x_i+
class GCN(nn.Module):
    def __init__(self, dim_in, n_cell, n_gcn, lamba):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.FloatTensor(abs(np.random.rand(dim_in).reshape([dim_in, 1]))))
        self.agg = nn.ModuleList()
        for i in range(n_gcn):
            #             self.agg.append(GCNConv(1,1))
            self.agg.append(AggNet())
        #         self.pred = L0Dense(n_cell,1,droprate_init=0.95,lamba=lamba,use_bias=True,weight_decay=0)
        self.pred = nn.Linear(n_cell, 1)

    def forward(self, x, edge, edge_weight):
        x = F.tanh((x@abs(self.parameter)).transpose(0,1).squeeze(-1))/x.shape[-1]
        # x = x.squeeze(-1).T
        #         x = x.mean(-1).transpose(0,1)
        x_raw = x
        xs = []
        for net in self.agg:
            x = net(x=x, edge_index=edge, edge_weight=edge_weight)
            # xs.append(x.transpose(0, 1))
        x = x.transpose(0, 1)
        #         print(x.shape)
        #         x = x+ x_raw[:,:].transpose(0,1)
        #         print(x_raw.shape)
        out = x  # +x_raw.transpose(0,1)
        #         for i in range(len(self.agg)-1,len(self.agg)):
        #             out += xs[i]
        out = self.pred(out)
        return out