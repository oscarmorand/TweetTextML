import time
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate, train_test_split
import matplotlib.pyplot as plt
import numpy as np
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

model_complete_name = {
    "dt": "Decision Tree",
    "rf": "Random Forest"
}

gradient_color = ((255, 255), (255, 0), (255, 0))


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


def test_model(data, targets, is_evaluation_used, model, model_name):
    all_scores = ([], [])
    labels = []
    for evaluation_name in evaluations:
        if is_evaluation_used[evaluation_name]:
            start_time = time.time()
            print("We use", evaluation_name, "for the evaluation")
            scores_acc, scores_auc = evaluations[evaluation_name](data, targets, model)
            all_scores[0].append(scores_acc)
            all_scores[1].append(scores_auc)
            labels.append(evaluation_name + " with\n" + model_complete_name[model_name])

            print(model_complete_name[model_name], "Acc:", scores_acc)
            print(model_complete_name[model_name], "AUC:", scores_auc)
            print("Runtime:", time.time() - start_time,"\n")
    return all_scores, labels


def print_scores(acc_scores, auc_scores, all_labels):
    print(all_labels)
    print(acc_scores)
    print(auc_scores)

    x = np.arange(len(all_labels))
    print(x)
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, acc_scores, width, label='Accuracy')
    rects2 = ax.bar(x + width / 2, auc_scores, width, label='AUC')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by model and evaluation method')
    ax.set_xticks(x, labels=all_labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def build_models(data, targets, is_evaluation_used, models, is_model_used):
    acc_scores = []
    auc_scores = []
    all_labels = []
    for model_name in models:
        if is_model_used[model_name]:
            print("Let's build a", model_complete_name[model_name], "model!\n")
            scores, labels = test_model(data, targets, is_evaluation_used, models[model_name], model_name)
            for acc_score in scores[0]:
                acc_scores.append(acc_score)
            for auc_score in scores[1]:
                auc_scores.append(auc_score)
            for label in labels:
                all_labels.append(label)
            print("End of the evaluations for the",model_complete_name[model_name], "model\n")
    print_scores(acc_scores, auc_scores, all_labels)