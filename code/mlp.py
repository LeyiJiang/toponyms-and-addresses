import os
from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        sequence_length = int(os.getenv('SEQUENCE_LENGTH'))
        vector_dim = int(os.getenv('EMBEDDING_DIM'))
        self.fc1 = nn.Linear(vector_dim * 2 * sequence_length, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-1)
        input = input.view(input.size(0), -1)
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        return output
