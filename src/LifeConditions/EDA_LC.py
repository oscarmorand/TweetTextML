import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Parsing_LC as parsing


# Import Data

header = []
data = []

#############################################################################
#
# Exploratory Data Analysis
#
#####################

colors = ['yellow', 'blue']


def std_list(list):
    return np.std(list) / 4


def show_all_bool_bar_plots(features, length):
    rows = int((len(features) - 1) / length) + 1
    fig, axs = plt.subplots(rows, length)

    i = 0
    for feature in features:
        target_false = [row[1][feature] for row in data if (row[0] == '0')]
        target_true = [row[1][feature] for row in data if (row[0] == '1')]

        bars = [np.mean(target_false), np.mean(target_true)]
        std = [std_list(target_false), std_list(target_true)]

        r = np.arange(len(bars))
        bar_width = 0.3

        y, x = int(i/length), int(i % length)
        axs[y, x].bar(r, bars, width=bar_width, color=colors, edgecolor='black', yerr=std)
        axs[y, x].set_xticks([r for r in range(len(bars))], ['No', 'Yes'])
        axs[y, x].set_xlabel("Depressed")
        axs[y, x].set_ylabel(header[feature])

        i += 1


def depressed_by_two_features(feature1, feature2):
    plt.figure()
    values = (([], []), ([], []))
    for row in data:
        values[int(row[1][feature1])][int(row[1][feature2])].append(float(row[0]))
    mean_values = [[0,0],[0,0]]
    std_values = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            mean_values[i][j] = np.mean(values[i][j])
            std_values[i][j] = std_list(values[i][j])

    index = np.arange(2)
    bar_width = 0.35
    plt.title("Depression by "+header[feature1]+" and "+header[feature2])
    plt.bar(index, mean_values[0], bar_width, color='orange', yerr=std_values[0], label='0')
    plt.bar(index+bar_width, mean_values[1], bar_width, color='blue', yerr=std_values[1], label='1')
    plt.xticks(index + bar_width / 2, ('0', '1'))
    plt.xlabel(header[feature2])
    plt.ylabel("Depressed")
    plt.legend(title=header[feature1])


def give_values_dict(feature):
    values_dict = {}
    for row in data:
        if float(row[0]) == 1.:
            if row[1][feature] in values_dict:
                values_dict[row[1][feature]] += 1
            else:
                values_dict[row[1][feature]] = 1
    return values_dict


def show_all_nbr_depressed_plot(features, length):
    rows = int((len(features)-1) / length) + 1
    fig, axs = plt.subplots(rows, length)
    i = 0
    for (feature, v_step) in features:
        values_dict = give_values_dict(feature)
        y, x = int(i / length), int(i % length)
        axs[y, x].set_xlabel(header[feature])
        axs[y, x].set_ylabel("Number of depressed people")
        if v_step:
            axs[y, x].set_title("Number of depressed people depending on " + header[feature] + " (using grouping with a step of " + str(v_step) + ")")
            max_value = max(values_dict)
            group_values = [0] * (int(max_value / v_step) + 1)
            for value in values_dict:
                index = int(value / v_step)
                group_values[index] += values_dict[value]
            axs[y, x].bar(list(range(0, (int(max_value / v_step) + 1) * v_step, v_step)), group_values, (100 / (v_step + 1)))
        else:
            axs[y, x].set_title("Number of depressed people depending on " + header[feature])
            def takeFirst(elem):
                return elem[0]
            data_list = []
            for value in values_dict:
                data_list.append((value, values_dict[value]))
            data_list.sort(key=takeFirst)
            axs[y, x].plot([row[0] for row in data_list], [row[1] for row in data_list])
        i += 1


if __name__ == '__main__':

    parsing.parse_lc(header, data)

    all_data = []
    for i in range(len(data)):
        all_data.append(data[i][1])
        all_data[i].append(int(data[i][0]))

    # =============== EDA ===============
    plt.figure()
    plt.title("Heatmap of the correlation matrix")
    pd_dataframe = pd.DataFrame(all_data, columns=header)   # Create a dataframe object that can be used with pandas library functions
    correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
    heatmap = sns.heatmap(pd_dataframe.corr(), annot=True)

    show_all_bool_bar_plots([3, 5, 6, 13], 2)  # sex, age, number of children, education level, incoming salary

    show_all_nbr_depressed_plot([(3, None), (3, 10), (5, None)], 2)  # age, age(group of 10 years), nbr of children

    depressed_by_two_features(4, 2)  # married and sex

    plt.show()