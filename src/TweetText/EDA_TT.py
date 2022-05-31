import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import nltk
import Parsing_TT as parsing

# remove temporary if you get the error "NLTK stop words not found", you only need to download it once
# nltk.download('stopwords')


#############################################################################
#
# Load Data
#
#####################

simplified_header = []
data = []
target = []
data_with_target = []

parsing.parse_tt(simplified_header, data, target, data_with_target)

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
stopwords = nltk.corpus.stopwords.words('english')
print("Stopwords in english: ",stopwords)
for word in stopwords:
    clean_text(word, " ", False, False)

print("After removing stopwords Data[0]: ", data[0])

tokenize_text()

print("After tokenization of words Data[0]: ", data[0])

stemming_text()

print("After stemming the words Data[0]: ", data[0])

removing_small_words(3)

print("After removing small words Data[0]: ", data[0])

