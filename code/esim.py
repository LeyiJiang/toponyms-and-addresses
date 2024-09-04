import torch
from torch import nn
from torch.nn import functional as F
import os

class ESIM(nn.Module):
    def __init__(self):
        super(ESIM, self).__init__()
        
        self.embedding_dim = 64
        self.hidden_size = 128
        self.dropout = 0.5
        
        # 做batchnormalization之前需要先转换维度
        self.bn_embeds = nn.BatchNorm1d(self.embedding_dim)
        self.lstm1 = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)
        
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(self.dropout),
            nn.Linear(2, 2),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(2),
            nn.Dropout(self.dropout),
            nn.Linear(2, 1),
        )
        
    def forward(self, q1, q2):
        o1, _ = self.lstm1(q1)
        o2, _ = self.lstm1(q2)

        q1_align, q2_align = self.soft_attention_align(o1, o2)

        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity
        

    def soft_attention_align(self, q1, q2):
        attention = torch.matmul(q1, q2.transpose(1, 2))
        weight1 = F.softmax(attention, dim=-1)
        q1_align = torch.matmul(weight1, q2)
        weight2 = F.softmax(attention.transpose(1, 2), dim=-1)
        q2_align = torch.matmul(weight2, q1)

        return q1_align, q2_align
    
    def submul(self, q1, q2):
        mul = q1 * q2
        sub = q1 - q2
        return torch.cat([sub, mul], -1)
    
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat([p1, p2], 1)
