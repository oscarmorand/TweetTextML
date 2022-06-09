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
show_graph = True
use_second_dataset = False
n_desired = 1000
# Put -1 if all dataset is desired

is_model_used = {
    "dt": True,
    "rf": True,
    "gb": True,
    "ada": True,
    "nn": True
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

if __name__ == '__main__':

    parsing.parse_tt(header, data, n_desired)

    if use_second_dataset:
        parsing.merge_sentiment_tweet_dataset(data)

    print("======== Model Building ========")

    random.shuffle(data)
    targets = [row[0] for row in data]

    count_vectorizer = CountVectorizer(stop_words='english')
    cv = count_vectorizer.fit_transform([row[1] for row in data])

    ml.build_models(cv, targets, is_evaluation_used, models, is_model_used, show_graph)

