from os.path import exists
import Parsing_LC as parsing
from sklearn.feature_extraction.text import CountVectorizer
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import FinalProjectML.src.Models as ml
warnings.filterwarnings("ignore")

header = []
data = []

#############################################################################
#
# Model Building
#
#####################

rand_st = 1

is_model_used = {
    "dt" : True,
    "rf": True
}

is_evaluation_used = {
    "test_split": True,
    "cross_validation": True
}

models = {
    "dt": DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=rand_st),
    "rf": RandomForestClassifier(n_estimators=10, random_state=rand_st)
}


if __name__ == '__main__':

    print("Parsing dataset...")
    parsing.parse_lc(header, data)

    print("======== Model Building ========")

    random.shuffle(data)
    targets = [row[0] for row in data]

    print(data[0])
    print(targets[0])

    ml.build_models([row[1] for row in data], targets, is_evaluation_used, models, is_model_used)