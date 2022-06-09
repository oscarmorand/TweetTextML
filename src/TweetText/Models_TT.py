import Parsing_TT as parsing
import Preprocessing_TT as preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import random
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
use_second_dataset = True

is_model_used = {
    "dt": True,
    "rf": False
}

is_evaluation_used = {
    "test_split": True,
    "cross_validation": True
}

models = {
    "dt": DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=None, random_state=rand_st),
    "rf": RandomForestClassifier(n_estimators=100, random_state=rand_st)
}

n_desired = 1000
# Put -1 if all dataset is desired

if __name__ == '__main__':

    parsing.parse_tt(header, data, n_desired)

    if use_second_dataset:
        parsing.merge_sentiment_tweet_dataset(data)

    print("======== Model Building ========")

    random.shuffle(data)
    targets = [row[0] for row in data]

    count_vectorizer = CountVectorizer(stop_words='english')
    cv = count_vectorizer.fit_transform([row[1] for row in data])

    ml.build_models(cv, targets, is_evaluation_used, models, is_model_used, True)

