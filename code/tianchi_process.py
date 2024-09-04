import os
import json
from dotenv import load_dotenv
import preprocess.tok as tok
from utils.preprocess import generate_grah_from_sentences
from tqdm import tqdm

load_dotenv()

if not os.getcwd().endswith('graduate'):
    os.chdir(os.getenv('PROJECT_PATH'))
tianchi_raw_dir = os.path.join(os.getcwd(), 'data', 'tianchi', 'raw')
files = ['dev.txt', 'train.txt']
raw_sentences = []

print('Loading raw sentences...')

for file in files:
    file_path = os.path.join(tianchi_raw_dir, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 0:
                obj = json.loads(line)
                raw_sentences.append(obj['sentence1'])

print('Done.')

print('Tokenizing raw sentences...')
sentences = []
for sentence in tqdm(raw_sentences, desc='Tokenizing'):
    words = tok.tokenize(sentence)
    sentences.append(words)
print('Done.')

print('Training word2vec model...')
word2vec_path = os.path.join(os.getcwd(), 'model', 'word2vec', 'word2vec_tianchi')
tok.trainWord2Vec(word2vec_path, sentences)
print('Done.')

print('Training node2vec model...')
node2vec_path = os.path.join(os.getcwd(), 'model', 'node2vec', 'node2vec_tianchi')
g = generate_grah_from_sentences(sentences)
tok.trainNode2Vec(node2vec_path, g)
print('Done.')
