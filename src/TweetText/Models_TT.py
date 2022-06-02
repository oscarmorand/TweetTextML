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

def test_split(model):
    X_train, X_test, y_train, y_test = train_test_split(cv, targets, test_size=.2, stratify=targets,
                                                        random_state=rand_st)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return accuracy_score(prediction, y_test), roc_auc_score(prediction, y_test)

def cross_validation(model):
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    scores = cross_validate(estimator=model, X=cv, y=targets, scoring=scorers, cv=n_cv)
    return scores['test_Accuracy'].mean(), scores['test_roc_auc'].mean()

rand_st = 1
is_model_used = {
    "rf": True
}
models = {
    "rf": RandomForestClassifier(n_estimators=10, random_state=rand_st)
}
model_complete_name = {"rf": "Random Forest"}

is_evaluation_used = {
    "test_split": True,
    "cross_validation" : True
}
evaluations = {
    "test_split": test_split,
    "cross_validation": cross_validation
}
n_cv = 5


def test_model(model):
    for evaluation_name in evaluations:
        if is_evaluation_used[evaluation_name]:
            start_time = time.time()
            print("We use", evaluation_name,"for the evaluation")
            scores_acc, scores_auc = evaluations[evaluation_name](model)

            print(model_complete_name[model_name], "Acc:", scores_acc)
            print(model_complete_name[model_name], "AUC:", scores_auc)
            print("Runtime:", time.time() - start_time,"\n")


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

    for model_name in models:
        if is_model_used[model_name]:
            print("Let's build a", model_complete_name[model_name], "model!\n")
            test_model(models[model_name])
            print("End of the evaluations for the",model_complete_name[model_name], "model\n")