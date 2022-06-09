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


def show_bool_bar_plot(feature):
    plt.figure()
    target_false = [row[1][feature] for row in data if (row[0] == '0')]
    target_true = [row[1][feature] for row in data if (row[0] == '1')]

    bars = [np.mean(target_false), np.mean(target_true)]
    std = [np.std(target_false), np.std(target_true)]
    colors = ['yellow', 'blue']

    r = np.arange(len(bars))
    bar_width = 0.3
    plt.bar(r, bars, width=bar_width, color=colors, edgecolor='black', yerr=std)
    plt.xticks([r for r in range(len(bars))], ['No', 'Yes'])
    plt.xlabel("Depressed")
    plt.ylabel(header[feature])


def show_targ_acc_feat_plot(feature):
    plt.figure()
    values_dict = {}
    for row in data:
        if row[1][feature] in values_dict:
            values_dict[row[1][feature]].append(float(row[0]))
        else:
            values_dict[row[1][feature]] = [float(row[0])]
    values = []
    means = []
    for value in values_dict:
        values.append(value)
        means.append(np.mean(values_dict[value]))
    values.sort()

    plt.plot(values, means)


if __name__ == '__main__':

    parsing.parse_lc(header, data)

    all_data = []
    for i in range(len(data)):
        all_data.append(data[i][1])
        all_data[i].append(int(data[i][0]))

    # =============== Correlation heatmap ===============
    plt.figure()
    pd_dataframe = pd.DataFrame(all_data, columns=header)   # Create a dataframe object that can be used with pandas library functions
    correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
    heatmap = sns.heatmap(pd_dataframe.corr(), annot=True)

    # =============== Bar Plots ===============
    show_bool_bar_plot(2)  # sex
    show_bool_bar_plot(3)  # age
    show_bool_bar_plot(5)  # number of children
    show_bool_bar_plot(6)  # education level
    show_bool_bar_plot(13)  # incoming salary

    show_targ_acc_feat_plot(3)  # age
    show_targ_acc_feat_plot(5)  # number of children

    plt.show()