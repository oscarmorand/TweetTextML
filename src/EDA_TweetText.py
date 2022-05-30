import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import seaborn as sns
import pandas as pd
import re
import nltk

# remove temporary if you get the error "NLTK stop words not found", you only need to download it once
# nltk.download('stopwords')

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
print('\n')

stopwords = nltk.corpus.stopwords.words('english')
print("Stopwords in english: ",stopwords)

#############################################################################
#
# Exploratory Data Analysis
#
#####################


def clean_text(before, after, del_all_word, sub_word):
    if del_all_word:
        before = before + "[A-Za-z:/.0-9]*"
    if not sub_word:
        before = " " + before + " "
    for i in range(len(data)):
        data[i][1] = re.sub(before, after, data[i][1])

def tokenize_text():
    for i in range(len(data)):
        data[i][1] = data[i][1].split()

def stemming_text():
    stemmer = nltk.stem.porter.PorterStemmer()
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            data[i][1][j] = stemmer.stem(data[i][1][j])

def removing_small_words(min_len):
    for i in range(len(data)):
        temp = []
        for j in range(len(data[i][1])):
            if len(data[i][1][j]) > min_len:
                temp.append(data[i][1][j])
        data[i][1] = temp

print("Before cleaning Data[0]: ", data[0])

# remove all characters that are not letters
clean_text("[^\(A-Za-z \)]", "", False, True)
# remove all words that begin with "http" in order to remove all links
clean_text("http", "", True, True)
# shorten all blank spaces that are longer than 1 space
clean_text("  *", " ", False, True)

print("After basic cleaning Data[0]: ", data[0])

# remove all english stopwords
for word in stopwords:
    clean_text(word, " ", False, False)

print("After removing stopwords Data[0]: ", data[0])

tokenize_text()

print("After tokenization of words Data[0]: ", data[0])

stemming_text()

print("After stemming the words Data[0]: ", data[0])

removing_small_words(3)

print("After removing small words Data[0]: ", data[0])

