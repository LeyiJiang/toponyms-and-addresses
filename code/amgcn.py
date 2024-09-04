import torch
from torch.nn import Dropout, ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, SAGPooling
import os

class AMGCN(torch.nn.Module):
    def __init__(self):
        super(AMGCN, self).__init__()
        max_query_length = int(os.getenv('SEQUENCE_LENGTH'))
        node_embedding_dim = int(os.getenv('EMBEDDING_DIM'))
        self.max_query_length = 2 * max_query_length
        self.dropout = Dropout(0.3)
        self.conv1 = GCNConv(node_embedding_dim, 300)
        self.conv2 = GCNConv(300, 300)
        self.conv3 = GCNConv(300, self.max_query_length)
        self.pool = global_mean_pool
        self.mlp = torch.nn.Linear(self.max_query_length, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = ReLU()(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = ReLU()(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = ReLU()(x)
        x = self.dropout(x)

        readout = self.pool(x, batch)
        readout = self.mlp(readout).squeeze(1)
        return readout