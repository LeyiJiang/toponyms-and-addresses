import torch
from bert import BERT
from gcn import GCN
from amgcn import AMGCN
from ablc import ABLC
from bilstm import BiLSTM
from esim import ESIM
from abcnn import ABCNN
from bimpm import BIMPM
from siagru import SiaGRU
from mlp import MLP
from dataset.geometric.tianchi_graph_dataset import TianchiGraphDataset
from dataset.geometric.deqing_graph_dataset_w2v import DeqingGraphDatasetW2V
from dataset.geometric.tianchi_graph_dataset_w2v import TianchiGraphDatasetW2V
from dataset.geometric.deqing_graph_dataset_notfidf import DeqingGraphDatasetNOTFIDF
from dataset.geometric.tianchi_graph_dataset_notfidf import TianchiGraphDatasetNOTFIDF
from dataset.tianchi_dataset import TianchiDataset
from dataset.geometric.deqing_graph_dataset import DeqingGraphDataset
from dataset.deqing_dataset import DeqingDataset
from sag import SAG
from sag_no_pool import SAG_NO_POOL

# Dictionary mapping model names to their classes
model_dict = {
    'SiaGRU': SiaGRU,
    'BIMPM': BIMPM,
    'ABCNN': ABCNN,
    'GCN': GCN,
    'AMGCN': AMGCN,
    'ABLC': ABLC,
    'BiLSTM': BiLSTM,
    'ESIM': ESIM,
    'MLP': MLP,
    'SAG': SAG,
    'SAG_NO_POOL': SAG_NO_POOL,
    'BERT': BERT,
}

# Factory method for creating modules
def NewModule(name) -> torch.nn.Module:
    if name in model_dict:
        return model_dict[name]()
    else:
        raise ValueError(f"Model {name} not recognized")

# Dictionaries mapping dataset names to their classes
dataset_dict = {
    'Tianchi': {
        True: TianchiGraphDataset,
        False: TianchiDataset
    },
    'Deqing': {
        True: DeqingGraphDataset,
        False: DeqingDataset
    },
    'Tianchi_W2V': {
        True: TianchiGraphDatasetW2V,
    },
    'Deqing_W2V': {
        True: DeqingGraphDatasetW2V,
    },
    'Tianchi_NOTFIDF': {
        True: TianchiGraphDatasetNOTFIDF,
    },
    'Deqing_NOTFIDF': {
        True: DeqingGraphDatasetNOTFIDF,
    }
}

# Factory method for creating datasets
def NewDataset(name, isGraph) -> torch.utils.data.Dataset:
    if name in dataset_dict and isGraph in dataset_dict[name]:
        return dataset_dict[name][isGraph]()
    else:
        raise ValueError(f"Dataset {name} with isGraph={isGraph} not recognized")