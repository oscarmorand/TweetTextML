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


def show_all_bool_bar_plots(features, length):
    rows = int(len(features)/length) + 1
    print(rows, length)
    fig, axs = plt.subplots(rows, length)

    i = 0
    for feature in features:
        target_false = [row[1][feature] for row in data if (row[0] == '0')]
        target_true = [row[1][feature] for row in data if (row[0] == '1')]

        bars = [np.mean(target_false), np.mean(target_true)]
        std = [np.std(target_false), np.std(target_true)]
        colors = ['yellow', 'blue']

        r = np.arange(len(bars))
        bar_width = 0.3

        y, x = int(i/length), int(i % length)
        axs[y, x].bar(r, bars, width=bar_width, color=colors, edgecolor='black', yerr=std)
        axs[y, x].set_xticks([r for r in range(len(bars))], ['No', 'Yes'])
        axs[y, x].set_xlabel("Depressed")
        axs[y, x].set_ylabel(header[feature])

        i += 1


def give_values_dict(feature):
    values_dict = {}
    for row in data:
        if float(row[0]) == 1.:
            if row[1][feature] in values_dict:
                values_dict[row[1][feature]] += 1
            else:
                values_dict[row[1][feature]] = 1
    return values_dict


def number_of_depressed(feature, v_step=None):
    plt.figure()
    values_dict = give_values_dict(feature)
    if v_step:
        max_value = max(values_dict)
        group_values = [0] * (int(max_value / v_step) + 1)
        for value in values_dict:
            index = int(value / v_step)
            group_values[index] += values_dict[value]
        plt.bar(list(range(0, (int(max_value / v_step) + 1) * v_step, v_step)), group_values, (100/(v_step+1)))
    else:
        def takeFirst(elem):
            return elem[0]
        data_list = []
        for value in values_dict:
            data_list.append((value, values_dict[value]))
        data_list.sort(key=takeFirst)
        plt.plot([row[0] for row in data_list], [row[1] for row in data_list])

    plt.xlabel(header[feature])
    plt.ylabel("Number of depressed people")


if __name__ == '__main__':

    parsing.parse_lc(header, data)

    all_data = []
    for i in range(len(data)):
        all_data.append(data[i][1])
        all_data[i].append(int(data[i][0]))

    # =============== EDA ===============
    plt.figure()
    pd_dataframe = pd.DataFrame(all_data, columns=header)   # Create a dataframe object that can be used with pandas library functions
    correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
    heatmap = sns.heatmap(pd_dataframe.corr(), annot=True)

    show_all_bool_bar_plots([2, 3, 5, 6, 13], 4)  # sex, age, number of children, education level, incoming salary

    number_of_depressed(3)  # number of depressed people by age
    number_of_depressed(3, 10)  # number of depressed people by age (group of 10 years)
    number_of_depressed(5)  # number of children

    plt.show()