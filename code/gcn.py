import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data):
        x_q, edge_index_q, batch_q = data.x_q, data.edge_index_q, data.x_q_batch
        x_r, edge_index_r, batch_r = data.x_r, data.edge_index_r, data.x_r_batch

        x_q = self.conv1(x_q, edge_index_q)
        x_q = F.relu(x_q)
        x_q = F.dropout(x_q)
        x_q = self.conv2(x_q, edge_index_q)
        x_q = F.relu(x_q)
        x_q = F.dropout(x_q)
        x_q = self.conv3(x_q, edge_index_q)
        x_q = F.relu(x_q)
        x_q = F.dropout(x_q)

        x_r = self.conv1(x_r, edge_index_r)
        x_r = F.relu(x_r)
        x_r = F.dropout(x_r)
        x_r = self.conv2(x_r, edge_index_r)
        x_r = F.relu(x_r)
        x_r = F.dropout(x_r)
        x_r = self.conv3(x_r, edge_index_r)
        x_r = F.relu(x_r)
        x_r = F.dropout(x_r)
        
        x_q = global_mean_pool(x_q, batch_q)
        x_r = global_mean_pool(x_r, batch_r)
        v = torch.cat((x_q, x_r), dim=1)

        return self.fc(v).squeeze(1)