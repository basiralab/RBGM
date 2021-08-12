import torch 
import torch.nn as nn 
from torch.nn.utils import weight_norm
from torch_geometric.nn import NNConv 
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch.nn import Linear, Sequential, ReLU
from data_utils import create_edge_index_attribute
from torch.nn.parameter import Parameter
from torch import mm as mm
from torch.nn import Tanh
from data_utils import create_edge_index_attribute
import torch.nn.functional as F


shape = torch.Size((1225, 1225))
hidden_state = torch.cuda.FloatTensor(shape)
torch.randn(shape, out=hidden_state)
#hidden_state = torch.rand(1225,1225)






class RNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(RNNCell, self).__init__()
        self.weight = nn.Linear(input_dim, hidden_dim, bias=True)
        self.weight_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.tanh = Tanh()
    
    def forward(self,x):
        global hidden_state
        h = hidden_state
        y = self.tanh(self.weight(x) + self.weight_h(h))
        hidden_state = y.detach()
        return y
        
        

def eucledian_distance(x):
    repeated_out = x.repeat(35,1,1)
    repeated_t = torch.transpose(repeated_out, 0, 1)
    diff = torch.abs(repeated_out - repeated_t)
    return torch.sum(diff, 2)


        
class GNN_1(nn.Module):
    def __init__(self):
        super(GNN_1, self).__init__()
        self.rnn = nn.Sequential(RNNCell(1,1225), ReLU())
        self.gnn_conv = NNConv(35, 35, self.rnn, aggr='mean', root_weight=True, bias = True)
        
    
    def forward(self, data):
        edge_index, edge_attr, _, _ = create_edge_index_attribute(data)
        x1 = F.relu(self.gnn_conv(data, edge_index, edge_attr))
        x1 = eucledian_distance(x1)
        return x1


def frobenious_distance(test_sample,predicted):
  diff = torch.abs(test_sample - predicted)
  dif = diff*diff
  sum_of_all = diff.sum()
  d = torch.sqrt(sum_of_all)
  return d
