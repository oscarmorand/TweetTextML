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

if __name__ == '__main__':

    parsing.parse_lc(header, data)

    all_data = []
    for i in range(len(data)):
        all_data.append(data[i][1])
        all_data[i].append(int(data[i][0]))

    pd_dataframe = pd.DataFrame(all_data, columns=header)   # Create a dataframe object that can be used with pandas library functions
    correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
    heatmap = sns.heatmap(pd_dataframe.corr(), annot=True)
    plt.show()