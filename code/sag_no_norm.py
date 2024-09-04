import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, global_mean_pool    
from torch_geometric.nn.aggr import SoftmaxAggregation
import os
from torch_geometric.utils import sort_edge_index

class SAG(nn.Module):
    def __init__(self):
        super(SAG, self).__init__()
        in_features = int(os.getenv('EMBEDDING_DIM'))
        hidden_nodes = 256
        self.conv1 = GCNConv(in_features, hidden_nodes)
        self.pool1 = SAGPooling(in_channels=hidden_nodes, ratio=0.8)
        self.readout1 = SoftmaxAggregation()
        
        self.conv2 = GCNConv(hidden_nodes, hidden_nodes)
        self.pool2 = SAGPooling(hidden_nodes, ratio=0.8)
        self.readout2 = SoftmaxAggregation()
        
        self.conv3 = GCNConv(hidden_nodes, hidden_nodes)  # input_channels is 256
        self.pool3 = SAGPooling(hidden_nodes, ratio=0.8)
        self.readout3 = SoftmaxAggregation()

        self.fc1 = nn.Linear(hidden_nodes * 6, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x_q, edge_index_q, batch_q = data.x_q, data.edge_index_q, data.x_q_batch
        x_r, edge_index_r, batch_r = data.x_r, data.edge_index_r, data.x_r_batch

        x_q = self.conv1(x_q, edge_index_q)
        x_q, edge_index_q, _, batch_q, _, _ = self.pool1(x_q, edge_index_q, batch=batch_q)
        x1_q = global_mean_pool(x_q, batch_q)
        x_r = self.conv1(x_r, edge_index_r)
        x_r, edge_index_r, _, batch_r, _, _ = self.pool1(x_r, edge_index_r, batch=batch_r)
        x1_r = global_mean_pool(x_r, batch_r)

        x_q = self.conv2(x_q, edge_index_q)
        x_q, edge_index_q, _, batch_q, _, _ = self.pool1(x_q, edge_index_q, batch=batch_q)
        x2_q = global_mean_pool(x_q, batch_q)
        x_r = self.conv2(x_r, edge_index_r)
        x_r, edge_index_r, _, batch_r, _, _ = self.pool1(x_r, edge_index_r, batch=batch_r)
        x2_r = global_mean_pool(x_r, batch_r)

        x_q = self.conv3(x_q, edge_index_q)
        x_q, edge_index_q, _, batch_q, _, _ = self.pool1(x_q, edge_index_q, batch=batch_q)
        x3_q = global_mean_pool(x_q, batch_q)
        x_r = self.conv3(x_r, edge_index_r)
        x_r, edge_index_r, _, batch_r, _, _ = self.pool1(x_r, edge_index_r, batch=batch_r)
        x3_r = global_mean_pool(x_r, batch_r)

        x_q = torch.cat([x1_q, x2_q, x3_q], dim=1)
        x_r = torch.cat([x1_r, x2_r, x3_r], dim=1)

        v = torch.cat((x_q, x_r), dim=1)
        v = self.fc1(v)
        v = self.dropout(v)
        v = self.fc2(v)
        v = self.dropout(v)
        v = self.fc3(v).squeeze(1)
        # v = F.log_softmax(v, dim=1).squeeze(1)

        return v