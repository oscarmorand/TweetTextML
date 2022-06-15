import time
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import numpy as np
import warnings
import FinalProjectML.src.LifeConditions.Preprocessing_LC as preprocessing
warnings.filterwarnings("ignore")


rand_st = 1
n_cv = 5

model_complete_name = {
    "dt": "Decision Tree",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "ada": "Ada Boosting",
    "nn": "Neural Network",
    "svm": "Support \nVector Machines"
}


# ============ Evaluation methods ============


def test_split(data, targets, model, param=None):
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=.2, stratify=targets,
                                                        random_state=rand_st)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    try:
        roc = roc_auc_score(prediction, y_test)
    except:
        roc = 0
    return accuracy_score(prediction, y_test), roc


def cross_validation(data, targets, model, param=None):
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
    scores = cross_validate(estimator=model, X=data, y=targets, scoring=scorers, cv=param)
    return scores['test_Accuracy'].mean(), scores['test_roc_auc'].mean()


evaluations = {
    "test_split": test_split,
    "cross_validation": cross_validation
}

# ============ Default models scores ============


def test_model(data, targets, is_evaluation_used, model, model_name):
    all_scores = ([], [])
    labels = []
    for evaluation_name in evaluations:
        if is_evaluation_used[evaluation_name]:
            start_time = time.time()
            print("\tWe use", evaluation_name, "for the evaluation")

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


def build_models(data, targets, is_evaluation_used, models, is_model_used, do_print):
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
            print("End of the evaluations for the", model_complete_name[model_name], "model\n")
    if do_print:
        print_scores(acc_scores, auc_scores, all_labels)
    return acc_scores, auc_scores, all_labels


# ============ Models scores with changing parameters ============


def cv_range(data, targets, models, cv_range):
    print("Let's test our models with a variation of number of cv folds")
    fig, axs = plt.subplots(2)
    fig.suptitle("Accuracy for each number of cross-validation folds")
    axs[0].set(xlabel='Number of cv folds', ylabel='Accuracy')
    axs[1].set(xlabel='Number of cv folds', ylabel='Runtime')
    for model in models:
        print("Variation of cv folds for "+model_complete_name[model[0]]+" model\n\twith", end='')
        acc, runtimes = [], []
        for cv in cv_range:
            print(cv,end=', ')
            start_time = time.time()
            acc.append(cross_validation(data, targets, model[1], cv)[0])
            runtimes.append(time.time()-start_time)
        print("folds")
        axs[0].plot(cv_range, acc, label=model_complete_name[model[0]])
        axs[0].legend(loc="upper left")
        axs[1].plot(cv_range, runtimes, label=model_complete_name[model[0]])
        axs[1].legend(loc="upper left")
    print()
    plt.show()


def test_model_parameters_range(data, targets, model, parameters, model_name):
    fig, axs = plt.subplots(2, len(parameters))
    fig.suptitle("Parameter variation for " + model_complete_name[model_name]+" model")
    i = 0
    for parameter_name in parameters:
        print("\tVariation of", parameter_name, "...")
        values = parameters[parameter_name]
        grid = ParameterGrid({parameter_name: values, 'random_state': [rand_st]})
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
    print("Let's test our models with a variation of the parameters")
    for model_name in models:
        if used_models[model_name]:
            print("Variations for", model_complete_name[model_name], "model...")
            test_model_parameters_range(data, targets, models[model_name], all_parameters[model_name], model_name)


def feat_select_range(data, targets, header, models, fs_range, fs_params):
    print("Let's test our models with a variation of the type of feature selection")
    fig, axs = plt.subplots(2)
    fig.suptitle("Accuracy for each type of feature selection")
    axs[0].set(xlabel='Feature selection type', ylabel='Accuracy')
    axs[1].set(xlabel='Feature selection type', ylabel='Runtime')
    for model in models:
        print("Variation of feature selection type for "+model_complete_name[model[0]]+" model")
        acc, runtimes = [], []
        for i in range(len(fs_range)):
            fs_type, fs_param = fs_range[i], fs_params[i]
            cp_data, cp_header = data.copy(), header.copy()
            if fs_type != 0:
                preprocessing.feature_selection(cp_data, targets, cp_header, fs_type, fs_param)
            start_time = time.time()
            acc.append(cross_validation(cp_data, targets, model[1], n_cv)[0])
            runtimes.append(time.time()-start_time)
        axs[0].plot(fs_range, acc, label=model_complete_name[model[0]])
        axs[1].plot(fs_range, runtimes, label=model_complete_name[model[0]])

    fs_names = ["No feature\nselection", "Stepwise Recursive\nBackwards Feature\nemoval", "Wrapper Select\nia model", "Univariate Feature\nSelection - Chi-squared"]
    axs[0].set_xticks([r for r in range(len(fs_range))], fs_names)
    axs[0].legend(loc="best")
    axs[1].set_xticks([r for r in range(len(fs_range))], fs_names)
    axs[1].legend(loc="best")
    print()
    plt.show()
