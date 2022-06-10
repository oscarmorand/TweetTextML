import matplotlib.pyplot as plt

import Parsing_TT as parsing
from sklearn.feature_extraction.text import CountVectorizer
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import FinalProjectML.src.Models as ml
warnings.filterwarnings("ignore")

header = []
data = []
data_with_target = []

#############################################################################
#
# Model Building
#
#####################

rand_st = 1

do_default_models_scores = False
do_parameter_range_models_scores = False
do_cv_range = True

cv_range = range(2, 5, 1)
show_graph = True

use_second_dataset = False
n_desired = 1000
# Put -1 if all dataset is desired

is_model_used = {
    "dt": True,
    "rf": True,
    "gb": True,
    "ada": True,
    "nn": False
}

is_evaluation_used = {
    "test_split": True,
    "cross_validation": True
}

models = {
    "dt": DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=rand_st),
    "rf": RandomForestClassifier(n_estimators=10, random_state=rand_st),
    "gb": GradientBoostingClassifier(n_estimators=10, loss='deviance', learning_rate=0.1, max_depth=3, min_samples_split=3, random_state=rand_st),
    "ada": AdaBoostClassifier(n_estimators=10, base_estimator=None, learning_rate=0.1, random_state=rand_st),
    "nn": MLPClassifier(activation='logistic', solver='adam',alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,), random_state=rand_st)
}

models_params= {
    "dt": {
        "criterion": ['gini', 'entropy'],
        "splitter": ['best', 'random'],
        "min_samples_split": [2, 3, 4],
        "min_samples_leaf": [1, 2],
        "max_features": ['auto', 'sqrt', 'log2'],
    },
    "rf": {
        "n_estimators": [20, 50, 100, 150, 200, 250],
        "criterion": ['gini', 'entropy', 'log_loss'],
    },
    "gb": {
        "loss": ['log_loss', 'deviance', 'exponential'],
        "learning_rate": [0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "criterion": ['friedman_mse', 'squared_error', 'mse'],
        "max_depth": [2, 3, 4],
        "min_sample_split": [1, 2, 3],
    },
    "ada": {
        "n_estimators": [20, 50, 100],
        "learning_rate": [0.5, 1.0, 2.0],
    },
    "nn": {
        "activation": ['identity', 'logistic', 'tanh', 'relu'],
        "solver": ['lbfgs', 'sgd', 'adam'],
        "alpha": [0.0001],
        "learning_rate": ['constant', 'invscaling', 'adaptive'],
        "max_iter": [100, 200, 500, 1000],
        "hidden_layer_sizes": [(10,), (20,), (50,), (100,)],
    }
}


if __name__ == '__main__':

    print("======== Load Data ========")

    parsing.parse_tt(header, data, n_desired)

    if use_second_dataset:
        parsing.merge_sentiment_tweet_dataset(data)

    print("======== Preprocessing ========")

    random.shuffle(data)
    targets = [row[0] for row in data]

    count_vectorizer = CountVectorizer(stop_words='english')
    vectors = count_vectorizer.fit_transform([row[1] for row in data])

    print("======== Feature Selection ========")

    print("======== Model Building ========")

    if do_default_models_scores:
        ml.build_models(vectors, targets, is_evaluation_used, models, is_model_used, show_graph)

    if do_cv_range:
        ml.cv_range(vectors, targets, [(model,models[model]) for model in models if is_model_used[model]], cv_range)

    if do_parameter_range_models_scores:
        ml.parameter_range(vectors, targets, models, is_model_used, models_params)

    plt.show()

