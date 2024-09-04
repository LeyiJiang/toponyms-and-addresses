"""
deqing_dataset module
---------------------

This module provides the DeqingDataset class for loading and processing Deqing data.

Classes:
    DeqingDataset: A PyTorch Geometric Dataset for Deqing data.

Example:
    To use the DeqingDataset class, you can do the following:

    from dataset.deqing_dataset import DeqingDataset

    dataset = DeqingDataset()

    print(dataset[0][0].x)
"""

import glob
from torch_geometric.data import Dataset
import networkx as nx
import json
import torch
import numpy as np
from gensim.models import Word2Vec
from torch_geometric.utils import from_networkx
from utils.utils import add_edge
import os.path as osp
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from .pair_data import PairData

def fn(s):
    return s

class DeqingGraphDatasetW2V(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.vectorizer = TfidfVectorizer(tokenizer=fn, lowercase=False, token_pattern=None)
        self.data = []
        if not os.getcwd().endswith('graduate'):
            os.chdir(os.getenv('PROJECT_PATH'))
        embbeding_path = os.path.join(os.getcwd(), 'model', 'word2vec', 'word2vec')
        self.model = Word2Vec.load(embbeding_path)
        self.sequence_length = int(os.getenv('SEQUENCE_LENGTH'))
        self.embbeding_dim = int(os.getenv('EMBEDDING_DIM'))
        for i in range(10):
            with open(os.path.join(self.raw_dir, 'final_data_{}.json'.format(i)), 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))

    @property
    def raw_file_names(self):
        return ['final_data_{}.json'.format(i) for i in range(10)]

    @property
    def raw_dir(self):
        # return a raw string of the path
        if not os.getcwd().endswith('graduate'):
            os.chdir(os.getenv('PROJECT_PATH'))
        return osp.join(os.getcwd(), 'data', 'deqing', 'raw')
    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        raw = self.data[idx]

        ref_graph = nx.Graph()
        query_graph = nx.Graph()
        add_edge(query_graph, ''.join(raw['q']), raw['q'], raw['tag_q'])
        add_edge(ref_graph, ''.join(raw['r']), raw['r'], raw['tag_r'])
        label =torch.tensor([raw['l']], dtype=torch.float32)
                
        corpus = [raw['q'], raw['r']]
        tfidf = self.vectorizer.fit_transform(corpus).toarray()

        # add node2vec embedding
        self.addNode2VecEmbedding(query_graph, tfidf[0], ''.join(raw['q']))
        self.addNode2VecEmbedding(ref_graph, tfidf[1], ''.join(raw['r']))

        data_q = from_networkx(query_graph)
        data_r = from_networkx(ref_graph)

        x_q = data_q.x
        x_r = data_r.x
        edge_index_q = data_q.edge_index
        edge_index_r = data_r.edge_index

        return PairData(x_q=x_q, edge_index_q=edge_index_q,
                        x_r=x_r, edge_index_r=edge_index_r,
                        y=label)

    def addNode2VecEmbedding(self, graph, tfidf, address):
        for node in graph.nodes:
            graph.nodes[node]['x'] = torch.zeros(self.embbeding_dim, dtype=torch.float32)
            if node == 'WORD':
                continue
            if self.model.wv.has_index_for(node):
                graph.nodes[node]['x'] = torch.tensor(self.model.wv[node], dtype=torch.float32)
            elif node == address:
                graph.nodes[node]['x'] = torch.tensor(np.pad(tfidf, (0, self.embbeding_dim-len(tfidf))), dtype=torch.float32)