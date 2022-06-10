import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np


rand_st = 1
n_cv = 5

model_complete_name = {
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "ada": "Ada Boosting",
    "nn": "Neural Network"
}