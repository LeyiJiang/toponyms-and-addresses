import torch
import torch.nn as nn
import torch.nn.functional as F


class SiaGRU(nn.Module):
    def __init__(self, hidden_size=64, num_layer=2):
        super(SiaGRU, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.embeds_dim = 64
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.gru = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)
        self.h0 = self.init_hidden((2 * self.num_layer, 1, self.hidden_size))
        self.h0.to(device)
        self.pred_fc = nn.Linear(256, 1)

    def init_hidden(self, size):
        h0 = nn.Parameter(torch.randn(size))
        nn.init.xavier_normal_(h0)
        return h0

    def forward_once(self, x):
        output, hidden = self.gru(x)
        return output
    
    def dropout(self, v):
        return F.dropout(v, p=0.2, training=self.training)

    def forward(self, s1, s2):
        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        p_encode = s1
        h_endoce = s2
        p_encode = self.dropout(p_encode)
        h_endoce = self.dropout(h_endoce)
        
        encoding1 = self.forward_once(p_encode)
        encoding2 = self.forward_once(h_endoce)
        sim = torch.exp(-torch.norm(encoding1 - encoding2, p=2, dim=-1, keepdim=True))
        x = self.pred_fc(sim.squeeze(dim=-1))
        probabilities = nn.functional.softmax(x, dim=-1)
        return probabilities