import re
import time
from tqdm import tqdm
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2

rand_st = 1


def remove_useless_features(data, header, useless_features):
    print(header)
    useful_features = []
    for i in range(len(header)-1):
        if header[i] not in useless_features:
            useful_features.append(i)
    print(useful_features)
    for x in useless_features:
        header.remove(x)
    for i in tqdm(range(len(data)), desc="Removing useless features...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        tmp = []
        for j in useful_features:
            tmp.append(data[i][j])
        data[i] = tmp
    print(header)
    print(data[0])
    print("Removing useless features done\n")


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

# ============ Feature selection ============


def feature_selection(data, targets, header, fs_type, fs_param):
    if fs_type == 1:
        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=rand_st)
        sel = RFE(clf, n_features_to_select=fs_param[0], step=.1)
    elif fs_type == 2:
        clf = GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, max_depth=3,
                                             min_samples_split=3, random_state=rand_st)
        sel = SelectFromModel(clf, prefit=False, threshold='mean',
                                  max_features=None)  # to select only based on max_features, set to integer value and set threshold=-np.inf
    else: #fs_type == 3 or something else
        sel = SelectKBest(chi2, k=fs_param[0])

    fit_mod = sel.fit(data, targets)
    sel_idx = fit_mod.get_support()

    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(header)-1):
        if sel_idx[i] == 1:  # Selected Features get added to temp header
            temp.append(header[i])
            temp_idx.append(i)
        else:  # Indexes of non-selected features get added to delete array
            temp_del.append(header[i])
    print('Selected ('+str(len(temp))+'):', temp)
    print('Removed features ('+str(len(temp_del))+'):', temp_del)
    print('Features (total/selected):', len(header)-1, len(temp))
    print('\n')

    remove_useless_features(data, header, temp_del)



