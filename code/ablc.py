import os
import torch
from torch import nn
import torch.nn.functional as F

class ABLC(nn.Module):
    def __init__(self):
        super(ABLC, self).__init__()
        embedding_dim = int(os.getenv('EMBEDDING_DIM'))
        self.lstm = nn.LSTM(embedding_dim, 32, bidirectional=True, batch_first=True)
        filters = 100
        kernel_size = 5 
        self.conv = nn.Conv1d(embedding_dim, filters, kernel_size)
        pool_size = 2
        self.padding_right = 2 * filters - embedding_dim
        self.max_pooling = nn.MaxPool1d(pool_size)
        self.attention = nn.MultiheadAttention(400, 1)

    def forward(self, Q, R):
        v1_q, _ = self.lstm(Q)
        v1_r, _ = self.lstm(R)

        shape = v1_q.shape
        padding_left = 0
        padding_right = 200 - shape[2] if shape[2] < 200 else 0
        padding_top = 0
        padding_bottom = 100 - shape[1] if shape[1] < 100 else 0
        v1_q = F.pad(v1_q, (padding_left, padding_right, padding_top, padding_bottom))
        v1_r = F.pad(v1_r, (padding_left, padding_right, padding_top, padding_bottom))

        Q_transpose = Q.transpose(1, 2)
        R_transpose = R.transpose(1, 2)
        v2_q = self.conv(Q_transpose)
        v2_r = self.conv(R_transpose)
        v2_q = self.max_pooling(v2_q)
        v2_r = self.max_pooling(v2_r)

        shape = v2_q.shape
        padding_left = 0
        padding_right = 200 - shape[2] if shape[2] < 200 else 0
        padding_top = 0
        padding_bottom = 256 - shape[1] if shape[1] < 256 else 0
        v2_q = F.pad(v2_q, (padding_left, padding_right, padding_top, padding_bottom))
        v2_r = F.pad(v2_r, (padding_left, padding_right, padding_top, padding_bottom))

        v_q = torch.cat((v2_q, v1_q), dim=-1)
        v_r = torch.cat((v2_r, v1_r), dim=-1)

        v_q = self.attention(v_q, v_q, v_q)[0]
        v_r = self.attention(v_r, v_r, v_r)[0]
        
        output = torch.abs(v_q - v_r)
        output = output.sum(dim=(-1, -2))

        return F.sigmoid(output).unsqueeze(1)