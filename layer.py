import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Parameter
import numpy as np

class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.normal_(self.weight.data, std=0.1)
        if self.bias is not None:
            nn.init.normal_(self.bias.data, std=0.1)

    def forward(self, input, adj):
        fea_tran = torch.mm(input, self.weight)
        agg = torch.sparse.mm(adj, fea_tran)
        out = torch.sparse.mm(adj, agg)

        if self.bias is not None:
            return out + self.bias
        else:
            return out

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
class BiGCN(nn.Module):
    def __init__(self, in_dim, out_dim, graph, bias=True):
        super(BiGCN, self).__init__()
        self.conv = GraphConvLayer(in_dim, out_dim, bias=bias)
        self.f1 = nn.Sequential(
            Linear(out_dim, 300),
            nn.ReLU(),
            Linear(300, 1024),
            nn.ReLU(),
            Linear(1024, out_dim),
            nn.ReLU()
        )
        self.f2 = nn.Sequential(
            Linear(out_dim, 300),
            nn.ReLU(),
            Linear(300, 1024),
            nn.ReLU(),
            Linear(1024, out_dim),
            nn.ReLU()
        )
        self.graph = graph


    def forward(self, all_emb, num_users, num_items):
        out = F.relu(self.conv(all_emb, self.graph))
        u, i = torch.split(out, [num_users,num_items])
        u = self.f1(u)
        i = self.f2(i)
        return u, i





