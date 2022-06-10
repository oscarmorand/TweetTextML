import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
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
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "ada": "Ada Boosting",
    "nn": "Neural Network"
}


def test_split(data, targets, model, param=None):
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=.2, stratify=targets,
                                                        random_state=rand_st)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return accuracy_score(prediction, y_test), roc_auc_score(prediction, y_test)


def cross_validation(data, targets, model, param=None):
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    scores = cross_validate(estimator=model, X=data, y=targets, scoring=scorers, cv=param)
    return scores['test_Accuracy'].mean(), scores['test_roc_auc'].mean()


evaluations = {
    "test_split": test_split,
    "cross_validation": cross_validation
}


def test_model(data, targets, is_evaluation_used, model, model_name, feat_select):
    all_scores = ([], [])
    labels = []
    for evaluation_name in evaluations:
        if is_evaluation_used[evaluation_name]:
            start_time = time.time()
            print("\tWe use", evaluation_name, "for the evaluation")

            '''
            sets = ParameterGrid( dictionnaire )
            for i in range(sets.__len__):
                classifier.set_params(**sets[i])
                
                joblib.dump() #pour enregistrer
            '''

            scores_acc, scores_auc = evaluations[evaluation_name](data, targets, model, n_cv)
            all_scores[0].append(scores_acc)
            all_scores[1].append(scores_auc)
            labels.append(evaluation_name + "\nwith\n" + model_complete_name[model_name])

            print("\t\t"+model_complete_name[model_name], "Acc:", scores_acc)
            print("\t\t"+model_complete_name[model_name], "AUC:", scores_auc)
            print("\t\tRuntime:", time.time() - start_time, "\n")
    return all_scores, labels


def print_scores(acc_scores, auc_scores, all_labels):
    x = np.arange(len(all_labels))
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


def build_models(data, targets, is_evaluation_used, models, is_model_used, do_print, feat_select):
    acc_scores = []
    auc_scores = []
    all_labels = []
    for model_name in models:
        if is_model_used[model_name]:
            print("Let's build a", model_complete_name[model_name], "model!\n")
            scores, labels = test_model(data, targets, is_evaluation_used, models[model_name], model_name, feat_select)
            for acc_score in scores[0]:
                acc_scores.append(acc_score)
            for auc_score in scores[1]:
                auc_scores.append(auc_score)
            for label in labels:
                all_labels.append(label)
            print("End of the evaluations for the", model_complete_name[model_name], "model\n")
    if do_print:
        print_scores(acc_scores, auc_scores, all_labels)
    return acc_scores, auc_scores, all_labels


def cv_range(data, targets, models, min, max, step):
    plt.figure()
    cvs = range(min, max, step)
    for model in models:
        acc = []
        for cv in cvs:
            acc.append(cross_validation(data, targets, model[1], cv)[0])
        plt.plot(cvs, acc, label=model_complete_name[model[0]])
    plt.legend(loc="upper left")


def test_model_parameters_range(data, targets, model, parameters, model_name):
    fig, axs = plt.subplots(2, len(parameters))
    fig.suptitle("Parameter variation of " + model_complete_name[model_name])
    #plt.subplots_adjust(0.04, 0.23, 0.99, 0.9)
    #wm = plt.get_current_fig_manager()
    #wm.window.state('zoomed')
    i = 0
    for parameter_name in parameters:
        print("\tVariation of", parameter_name, "...")
        values = parameters[parameter_name]
        grid = ParameterGrid({parameter_name: values})
        scores_acc, scores_auc, runtimes = [], [], []
        for j in range(len(grid)):
            model.set_params(**grid[j])
            start_time = time.time()
            score_acc, score_auc = cross_validation(data, targets, model, n_cv)
            scores_acc.append(score_acc)
            scores_auc.append(score_auc)
            runtimes.append(time.time()-start_time)

        axs[0, i].plot(values, scores_acc, label="Accuracy")
        axs[0, i].plot(values, scores_auc, label="AUC")
        axs[0, i].legend(loc="best")
        axs[0, i].set_title(parameter_name)

        axs[1, i].plot(values, runtimes, label="Runtimes")
        axs[1, i].legend(loc="best")

        i += 1
    plt.show()


def parameter_range(data, targets, models, used_models, all_parameters):
    for model_name in models:
        if used_models[model_name]:
            print("Variations for", model_complete_name[model_name], "model...")
            test_model_parameters_range(data, targets, models[model_name], all_parameters[model_name], model_name)
