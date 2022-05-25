import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas as pd
import re

#############################################################################
#
# Global parameters
#
#####################

target_idx = 0
feat_start = 1
feat_end = 5
important_feat = 5
n_desired = 100

#############################################################################
#
# Load Data
#
#####################

file1 = csv.reader(open('../../datasets/TweetText_Dataset.csv'), delimiter=',', quotechar='"')

#No Header Line in the dataset
header = ["target","id","date","flag","user","text"]
simplified_header = ["target","text"]

#Read data
data = []
target = []
all_data = []
i = 0
for row in file1:
    if i > n_desired-1:
        break

    # Load Target
    temp = []
    target.append(row[target_idx])               # Add the target on the target array
    temp.append(row[target_idx])

    # Load Data
    temp.append(row[important_feat])
    data.append(temp)
    full_temp = []
    for x in row:
        full_temp.append(x)
    all_data.append(full_temp)

    i += 1

#Test Print
print("Full Header ["+str(len(header))+"]:", header)
print("Simplified Header ["+str(len(simplified_header))+"]:", simplified_header, "\n")

print("Data[0]: ", data[0])
print("Target[0]: ", target[0])
print("AllData[0]: ", all_data[0],'\n')

print("str Data["+str(len(data))+"]")
print(str(len(target))+" targets, "+str(len(data))+" data", end=': ')
if len(target) == len(data):
    print("same lengths, OK")
else:
    print("different lengths, NOT OK")

#############################################################################
#
# Exploratory Data Analysis
#
#####################


def clean_text(before, after, allWord):
    if allWord:
        before = before+"[A-Za-z:/.0-9]*"
    for i in range(len(data)):
        data[i][1] = re.sub(before, after, data[i][1])


# remove all characters that are not letters
clean_text("[^\(A-Za-z \)]", "", False)
# remove all words that begin with "http" in order to remove all links
clean_text("http", "", True)
# shorten all blank spaces that are longer than 1 space
clean_text("  *", " ", False)

print("Data[0]: ", data[0])

'''
pd_dataframe = pd.DataFrame(data_with_target, columns=header)   # Create a dataframe object that can be used with pandas library functions
correlogram = pd_dataframe.corr()   # Create a correlation matrix with this dataframe
heatmap = sns.heatmap(pd_dataframe.corr())
plt.show()
'''