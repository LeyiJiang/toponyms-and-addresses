def levenshtein_distance(tokens1, tokens2):
    size_x = len(tokens1) + 1
    size_y = len(tokens2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]
    for x in range(size_x):
        matrix [x][0] = x
    for y in range(size_y):
        matrix [0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if tokens1[x-1] == tokens2[y-1]:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    return matrix[size_x - 1][size_y - 1]

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return len(s1.intersection(s2)) / len(s1.union(s2))

def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

import os
import json
from tqdm import tqdm

max_len = -1
min_len = 100
matches = 0
len_sum = 0
mismatches = 0
input_path = '../data/deqing_original'
edit_distance_matches = 0
edit_distance_mismatches = 0
jaccard_similarity_matches = 0
jaccard_similarity_mismatches = 0
hamming_distance_matches = 0
hamming_distance_mismatches = 0

for i in tqdm(range(10)):
    with open(os.path.join(input_path, f'final_data_{i}.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        for data in json_data:
            query = data['q']
            ref = data['r']
            label = data['l']
            len_sum += len(query) + len(ref)
            if len(query) > max_len:
                max_len = len(query)
            if len(query) < min_len:
                min_len = len(query)
            if len(ref) > max_len:
                max_len = len(ref)
            if len(ref) < min_len:
                min_len = len(ref)
            if label == 1:
                matches += 1
            else :
                mismatches += 1

            edit_dist = levenshtein_distance(query, ref)
            jaccard_dist = jaccard_similarity(query, ref)
            hamming_dist = hamming_distance(query, ref)
            if label == 1:
                edit_distance_matches += edit_dist
                jaccard_similarity_matches += jaccard_dist
                hamming_distance_matches += hamming_dist
            else:
                edit_distance_mismatches += edit_dist
                jaccard_similarity_mismatches += jaccard_dist
                hamming_distance_mismatches += hamming_dist

print(f'average_len: {len_sum / (matches + mismatches) * 2}')
print(f'max_len: {max_len}')
print(f'min_len: {min_len}')
print(f'matches: {matches}')
print(f'mismatches: {mismatches}')
print(f'edit_distance_matches: {edit_distance_matches / matches}')
print(f'edit_distance_mismatches: {edit_distance_mismatches / mismatches}')
print(f'edit_distance: {(edit_distance_matches + edit_distance_mismatches) / (matches + mismatches)}')
print(f'jaccard_similarity_matches: {jaccard_similarity_matches / matches}')
print(f'jaccard_similarity_mismatches: {jaccard_similarity_mismatches / mismatches}')
print(f'jaccard_similarity: {(jaccard_similarity_matches + jaccard_similarity_mismatches) / (matches + mismatches)}')
print(f'hamming_distance_matches: {hamming_distance_matches / matches}')
print(f'hamming_distance_mismatches: {hamming_distance_mismatches / mismatches}')
print(f'hamming_distance: {(hamming_distance_matches + hamming_distance_mismatches) / (matches + mismatches)}')


