from torch.utils.data import Dataset
import os
import json
import torch
import numpy as np
from gensim.models import Word2Vec

class DeqingDataset(Dataset):
    def __init__(self):
        if not os.getcwd().endswith('graduate'):
            os.chdir(os.getenv('PROJECT_PATH'))
        self.processed_dir = os.path.join(os.getcwd(), 'data', 'deqing', 'raw')
        self.train_data = []
        embbeding_path = os.path.join(os.getcwd(), 'model', 'word2vec', 'word2vec')
        self.model = Word2Vec.load(embbeding_path)
        self.sequence_length = int(os.getenv('SEQUENCE_LENGTH'))
        self.embbeding_dim = int(os.getenv('EMBEDDING_DIM'))
        for i in range(10):
            with open(os.path.join(self.processed_dir, 'final_data_{}.json'.format(i)), 'r', encoding='utf-8') as f:
                self.train_data.extend(json.load(f))
    
    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        item = self.train_data[idx]
        label = item['l']
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        q = np.array([self.model.wv[word] if word in self.model.wv else [0] * self.embbeding_dim for word in item['q']])
        r = np.array([self.model.wv[word] if word in self.model.wv else [0] * self.embbeding_dim for word in item['r']])
        q = torch.tensor(q, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.float32)
        # extend q and r to sequence_length
        q = torch.cat((q, torch.zeros(self.sequence_length - len(q), self.embbeding_dim)))
        r = torch.cat((r, torch.zeros(self.sequence_length - len(r), self.embbeding_dim)))

        return q, r, label