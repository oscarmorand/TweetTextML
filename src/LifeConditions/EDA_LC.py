import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import Parsing_LC as parsing


# Import Data

header = []
data = []
target = []
data_with_target = []

parsing.parse_lc(header, data, target, data_with_target)

#############################################################################
#
# Exploratory Data Analysis
#
#####################

print(header)
print(data_with_target[0])

pd_dataframe = pd.DataFrame(data_with_target, columns=header)   # Create a dataframe object that can be used with pandas library functions
correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
heatmap = sns.heatmap(pd_dataframe.corr())
plt.show()