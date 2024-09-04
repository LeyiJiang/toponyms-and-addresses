import torch
from torch import nn
import os

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        hidden_nodes = 128
        self.lstm = nn.LSTM(int(os.getenv('EMBEDDING_DIM')), hidden_nodes, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_nodes * 4, 1)

    def forward(self, Q, R):
        encoding_q, _ = self.lstm(Q)
        encoding_r, _ = self.lstm(R)

        encoding_q = torch.max(encoding_q, dim=1)[0]
        encoding_r = torch.max(encoding_r, dim=1)[0]

        v = torch.cat((encoding_r, encoding_q), dim=-1)

        # MLP prediction layer, reduce to 2 classes
        v = self.fc(v)
        return v