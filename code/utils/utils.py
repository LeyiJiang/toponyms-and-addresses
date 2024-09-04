import networkx as nx
import csv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import numpy as np

def add_edge(G, address, words, tags):
    """add an edge to G with given words

    Args:
    - G (nx.Graph): The edge owner graph.
    - address (str): The address of the given words.
    - words (list of str): A tokenized address.
    - tags (list of str): POS tags for words.

    Returns:
    - nx.Graph: G with the newly added edge.

    """
    intersection_tags = { 'p', 'c' }
    number_tags = { 'm' }
    for idx, tag in enumerate(tags):
        if idx == len(tags) - 1:
            nx.add_path(G, words)
            G.add_edge(words[idx], address)
        elif tag in intersection_tags:
            nx.add_path(G, words[:idx])
            G.add_edge(words[idx+1], address)
            G.add_edge(words[idx-1], address)
            break
        elif tag in number_tags:
            nx.add_path(G, words[:idx])
            G.add_edge(words[idx-1], address)
            break

# The following functions are used for logging metrics during training

def log_metrics(metrics, filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(metrics)

def write_headers(headers, filename):
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'x') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

def calc_metrics(preds, labels):
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='binary')
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='macro')
    return [accuracy, recall, precision, f1]

# here's k-shot dataset generation code
def get_balanced_loader(dataset, k, isGraph=False):
    # Get the indices for each class
    class0_indices = []
    class1_indices = []

    if not isGraph:
        for i, (q, r, y) in enumerate(dataset):
            if len(class0_indices) == k and len(class1_indices) == k:
                break
            if y == 0 and len(class0_indices) < k:
                class0_indices.append(i)
            elif y == 1 and len(class1_indices) < k:
                class1_indices.append(i)
    else:
        for i, data in enumerate(dataset):
            if len(class0_indices) == k and len(class1_indices) == k:
                break
            if data.y == 0 and len(class0_indices) < k:
                class0_indices.append(i)
            elif data.y == 1 and len(class1_indices) < k:
                class1_indices.append(i)

    # Combine the selected indices
    combined_indices = np.concatenate([class0_indices, class1_indices])

    # Create a Subset using the combined indices
    subset = Subset(dataset, combined_indices)

    return subset