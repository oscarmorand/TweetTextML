import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, train_test_split
import random
import warnings
warnings.filterwarnings("ignore")

#############################################################################
#
# Model Building
#
#####################

rand_st = 1
n_cv = 5

model_complete_name = {"rf": "Random Forest"}


def test_split(data, targets, model):
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=.2, stratify=targets,
                                                        random_state=rand_st)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return accuracy_score(prediction, y_test), roc_auc_score(prediction, y_test)


def cross_validation(data, targets, model):
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    scores = cross_validate(estimator=model, X=data, y=targets, scoring=scorers, cv=n_cv)
    return scores['test_Accuracy'].mean(), scores['test_roc_auc'].mean()


evaluations = {
    "test_split": test_split,
    "cross_validation": cross_validation
}

models = {
    "rf": RandomForestClassifier(n_estimators=10, random_state=rand_st)
}


def test_model(data, targets, is_evaluation_used, model, model_name):
    for evaluation_name in evaluations:
        if is_evaluation_used[evaluation_name]:
            start_time = time.time()
            print("We use", evaluation_name,"for the evaluation")
            scores_acc, scores_auc = evaluations[evaluation_name](data, targets, model)

            print(model_complete_name[model_name], "Acc:", scores_acc)
            print(model_complete_name[model_name], "AUC:", scores_auc)
            print("Runtime:", time.time() - start_time,"\n")


def build_models(data, targets, is_evaluation_used, is_model_used):
    for model_name in models:
        if is_model_used[model_name]:
            print("Let's build a", model_complete_name[model_name], "model!\n")
            test_model(data, targets, is_evaluation_used, models[model_name], model_name)
            print("End of the evaluations for the",model_complete_name[model_name], "model\n")