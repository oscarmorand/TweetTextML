import re
import time
from tqdm import tqdm
import sys


def remove_useless_features(data, header, useful_features):
    new_header = []
    print("\nUseful features: ", end='')
    for x in useful_features:
        new_header.append(header[x])
        print(header[x]+' ', end='')
    print()
    for i in tqdm(range(len(data)), desc="Removing useless features...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        tmp = []
        for j in useful_features:
            tmp.append(data[i][1][j])
        data[i][1] = tmp
    print("Removing useless features done\n")
    return new_header


def get_features_indexes(header, features):
    features_indexes = []
    for index in range(len(header)):
        if header[index] in features:
            features_indexes.append(index)
    return features_indexes


def normalization(data, useful_features, normalization_table):
    features_indexes = get_features_indexes(useful_features, normalization_table)
    print("Indexes of features to normalize:",features_indexes)
    length = len(features_indexes)
    max_of_features = [0] * length
    for i in tqdm(range(len(data)), desc="Normalizing 1/2...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        for j in range(length):
            if data[i][1][features_indexes[j]] > max_of_features[j]:
                max_of_features[j] = data[i][1][features_indexes[j]]
    print("Maximum of each feature to normalize:",max_of_features)
    for i in tqdm(range(len(data)), desc="Normalizing 2/2...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        for j in range(len(features_indexes)):
            data[i][1][features_indexes[j]] /= max_of_features[j]
    print(data[0][1])



