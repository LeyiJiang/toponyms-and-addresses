"""
This module provides functions for text preprocessing using HanLP library.

Functions:
- tokenize: Tokenizes text using HanLP tokenizer.
- tag: Tags words using HanLP part-of-speech tagger.
- trainWord2Vec: Trains a word2vec model.
- trainNode2Vec: Trains a node2vec model.
"""

from json import load
import hanlp
from gensim.models import Word2Vec
from node2vec import Node2Vec
import os
from dotenv import load_dotenv

load_dotenv()
if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))
tok_path = os.path.join(os.getcwd(), 'model', 'tok')
tok = hanlp.load(tok_path)
tokenizer = tok
tag = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)
    
def tokenize(text) -> list:
    """
    Tokenizes text using HanLP tokenizer.

    Args:
    - text (str): Input text to be tokenized.

    Returns:
    - list: List of tokens.
    """
    return tok(text)

def tags(words) -> list:
    """
    Tags words using HanLP part-of-speech tagger.

    Args:
    - words (list): List of words to be tagged.

    Returns:
    - list: List of tagged words.
    """
    return tag(words)

def trainWord2Vec(model_path, sentences) -> Word2Vec:
    """
    Trains a Word2Vec model using the given sentences.

    Args:
    - model_path (str): The path where the trained model will be saved.
    - sentences (list of list of str): The sentences used for training the model. Each sentence is a list of words.

    Returns:
    - Word2Vec: The trained Word2Vec model.
    """
    load_dotenv()
    if os.path.exists(model_path):
        os.remove(model_path)
    word2vec_dim = int(os.getenv('EMBEDDING_DIM'))
    model = Word2Vec(sentences, vector_size=word2vec_dim, window=5, min_count=1, workers=4)
    model.save(model_path)
    return model
    

def trainNode2Vec(model_path, graph) -> Node2Vec:
    """
    Trains a Node2Vec model using the given graph.

    Args:
    - model_path (str): The path where the trained model will be saved.
    - graph (networkx.Graph): The graph used for training the model.

    Returns:
    - Node2Vec: The trained Node2Vec model.
    """

    if os.path.exists(model_path):
        os.remove(model_path)
    load_dotenv()
    node2vec_dim = int(os.getenv('EMBEDDING_DIM'))
    node2vec = Node2Vec(graph, dimensions=node2vec_dim, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit()
    model.save(model_path)
    return model