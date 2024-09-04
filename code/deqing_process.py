"""
deqing_process.py

This script is used for processing raw data files in the Deqing dataset.

The logic is slightly different from the Tianchi dataset, because the Deqing dataset is already tokenized.

Imports:
- os: Provides functions for interacting with the operating system.
- json: Used for parsing and manipulating JSON data.
- dotenv: Used for loading environment variables from a .env file.
- preprocess.tok: A module for tokenizing text data.
- utils.utils: A module that contains utility functions, including a function for generating a graph from sentences.
- tqdm: A module for creating progress bars.


Variables:
- files: A list of filenames to be processed. The filenames are in the format 'final_data_{i}.json', where {i} is a number from 0 to 9.
- sentences: An empty list that will be filled with sentences from the data files.

Process:
- The script first loads the environment variables from a .env file.
- It then prints a message indicating that it's starting to load tokenized sentences.
"""
import os
import json
from dotenv import load_dotenv
import preprocess.tok as tok
from utils.preprocess import generate_grah_from_sentences

load_dotenv()

if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))
deqing_raw_dir = os.path.join(os.getcwd(), 'data', 'deqing', 'raw')
files = ['final_data_{}.json'.format(i) for i in range(10)]
sentences = []

print('Loading tokenized sentences...')

for file in files:
    file_path = os.path.join(deqing_raw_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        json_arr = json.load(f)
        for obj in json_arr:
            sentences.append(obj['r'])

print('Done.')

print('Training word2vec model...')
word2vec_path = os.path.join(os.getcwd(), 'model', 'word2vec', 'word2vec')
tok.trainWord2Vec(word2vec_path, sentences)
print('Done.')

print('Training node2vec model...')
g = generate_grah_from_sentences(sentences)
node2vec_path = os.path.join(os.getcwd(), 'model', 'node2vec', 'node2vec-1-1')
tok.trainNode2Vec(node2vec_path, g)
print('Done.')
