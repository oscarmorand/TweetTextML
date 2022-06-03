import time
from os.path import exists
import Parsing_TT as parsing
import Preprocessing_TT as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, train_test_split
import random
import warnings
import FinalProjectML.src.Models as models
warnings.filterwarnings("ignore")

header = []
data = []
targets = []
data_with_target = []

#############################################################################
#
# Model Building
#
#####################

is_model_used = {
    "rf": True
}

is_evaluation_used = {
    "test_split": True,
    "cross_validation" : True
}


if __name__ == '__main__':

    if not exists("../../../datasets/TweetText_Clean_Dataset.csv"):
        print("Clean dataset not found, parsing raw dataset...")
        parsing.parse_raw_tt(header, data, targets, data_with_target)
        print("Parsing complete\n======== CLEANING ========")
        preprocessing.clean_dataset(data)
        print("Cleaning complete, saving the clean dataset...")
        parsing.save_clean_tt(header, data)
        print("Saving complete\n")
    else:
        print("Clean dataset found, parsing clean dataset...")
        parsing.parse_clean_tt(header, data)
        print("Parsing complete\n")

    print("======== Model Building ========")

    random.shuffle(data)
    targets = [row[0] for row in data]

    count_vectorizer = CountVectorizer(stop_words='english')
    cv = count_vectorizer.fit_transform([row[1] for row in data])

    models.build_models(cv, targets, is_evaluation_used, is_model_used)

