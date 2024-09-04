import preprocess.tok as tok
from tqdm import tqdm
import networkx as nx
from .utils import add_edge

def generate_grah_from_sentences(sentences) -> nx.Graph:
    """
    Generates a graph from the given sentences.

    Args:
    - sentences (list of list of str): The sentences used for generating the graph. Each sentence is a list of words.

    Returns:
    - nx.Graph: The generated graph.
    """
    G = nx.Graph()

    for words in tqdm(sentences):
        tags = tok.tags(words)
        add_edge(G, ''.join(words), words, tags)

    return G