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

class TianchiGraphDatasetNOTFIDF(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.vectorizer = TfidfVectorizer(tokenizer=fn, lowercase=False, token_pattern=None)
        self.data = []
        embbeding_path = os.getenv('TIANCHI_NODE2VEC_PATH')
        embbeding_path = os.path.join(os.getenv('PROJECT_PATH'), 'model', 'node2vec', 'node2vec_tianchi')
        self.model = Word2Vec.load(embbeding_path)
        self.sequence_length = int(os.getenv('SEQUENCE_LENGTH'))
        self.embbeding_dim = int(os.getenv('EMBEDDING_DIM'))
        for i in range(10):
            with open(os.path.join(self.raw_dir, 'tianchi_{}.json'.format(i)), 'r', encoding='utf-8') as f:
                self.data.extend(json.load(f))

    @property
    def raw_file_names(self):
        return ['tianchi_{}.json'.format(i) for i in range(10)]

    @property
    def raw_dir(self):
        # return a raw string of the path
        if not os.getcwd().endswith('graduate'):
            os.chdir(os.getenv('PROJECT_PATH'))
        return os.path.join(os.getcwd(), 'data', 'tianchi', 'raw')
    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        raw = self.data[idx]

        ref_graph = nx.Graph()
        query_graph = nx.Graph()
        add_edge(query_graph, ''.join(raw['que']), raw['que'], raw['que_tag'])
        add_edge(ref_graph, ''.join(raw['ref']), raw['ref'], raw['ref_tag'])
        label =torch.tensor([raw['ans']], dtype=torch.float32)
                
        corpus = [raw['que'], raw['ref']]
        tfidf = self.vectorizer.fit_transform(corpus).toarray()

        # add node2vec embedding
        self.addNode2VecEmbedding(query_graph, tfidf[0], ''.join(raw['que']))
        self.addNode2VecEmbedding(ref_graph, tfidf[1], ''.join(raw['ref']))

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