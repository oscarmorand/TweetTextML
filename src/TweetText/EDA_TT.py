import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import nltk
import Parsing_TT as parsing
from threading import Lock, Thread
import multiprocessing as mp
import time
from tqdm import tqdm
from os.path import exists
import sys

# remove temporary if you get the error "NLTK stop words not found", you only need to download it once
# nltk.download('stopwords')

simplified_header = []
data = []
target = []
data_with_target = []

#############################################################################
#
# Exploratory Data Analysis
#
#####################


def clean_text(before, after, del_all_word, sub_word, bar_text=None):
    if del_all_word:
        before = before + "[A-Za-z:/.0-9]*"
    if not sub_word:
        before = " " + before + " "
    if bar_text:
        for i in tqdm(range(len(data)), desc=bar_text, ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
            data[i][1] = re.sub(before, after, data[i][1])
    else:
        for i in range(len(data)):
            data[i][1] = re.sub(before, after, data[i][1])


def tokenize_text():
    for i in tqdm(range(len(data)), desc="Tokenize words...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        data[i][1] = data[i][1].split()


def stemming_text():
    stemmer = nltk.stem.porter.PorterStemmer()
    for i in tqdm(range(len(data)), desc="Stemming text...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        for j in range(len(data[i][1])):
            data[i][1][j] = stemmer.stem(data[i][1][j])


def removing_small_words(min_len):
    for i in tqdm(range(len(data)), desc="Removing small words...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        temp = []
        for j in range(len(data[i][1])):
            if len(data[i][1][j]) > min_len:
                temp.append(data[i][1][j])
        data[i][1] = temp


def put_back_together_words():
    for i in tqdm(range(len(data)), desc="Putting the words back together...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        text = ""
        for word in data[i][1]:
            text += word + " "
        text = text[:-1]
        data[i][1] = text


def clean_dataset():
    print("Before cleaning Data[0]: ", data[0])
    start_time = time.time()

    # remove all characters that are not letters
    clean_text("[^\(A-Za-z \)]", "", False, True, "Removing non-letters characters...")
    print("After removing non-letters characters Data[0]: ", data[0], '\n')

    # remove all words that begin with "http" in order to remove all links
    clean_text("http", "", True, True, "Removing links...")
    print("After removing links Data[0]: ", data[0], '\n')

    # shorten all blank spaces that are longer than 1 space
    clean_text("  *", " ", False, True, "Removing large blank spaces...")
    print("After removing large blank spaces Data[0]: ", data[0],'\n')

    # remove all english stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    n_stopwords = len(stopwords)
    for i in tqdm(range(n_stopwords), desc="Removing stopwords...", ascii=False, ncols=100, leave=True, file=sys.stdout):
        stopword = stopwords[i]
        clean_text(stopword, " ", False, False)
    print("After removing stopwords Data[0]: ", data[0], '\n')

    tokenize_text()
    print("After tokenization of words Data[0]: ", data[0], '\n')

    stemming_text()
    print("After stemming the words Data[0]: ", data[0], '\n')

    removing_small_words(3)
    print("After removing small words Data[0]: ", data[0], '\n')

    put_back_together_words()
    print("After putting back together words Data[0]: ", data[0], '\n')


if __name__ == '__main__':

    if not exists("../../datasets/TweetText_Clean_Dataset.csv"):
        print("Clean dataset not found, parsing raw dataset...")
        parsing.parse_raw_tt(simplified_header, data, target, data_with_target)
        print("Parsing complete\n======== CLEANING ========")
        clean_dataset()
        print("Cleaning complete, saving the clean dataset...")
        parsing.save_clean_tt(simplified_header, data)
        print("Saving complete\n")
    else:
        print("Clean dataset found, parsing clean dataset...")
        parsing.parse_clean_tt(simplified_header, data)
        print("Parsing complete\n")

    print("======== EDA ========")

    negative_tweets = []
    positive_tweets = []
    for line in data:
        if line[0] == '0':
            negative_tweets.append(line[1])
        elif line[0] == '4':
            positive_tweets.append(line[1])

    print("The first line of negative tweets:", negative_tweets[0])
    if len(positive_tweets) > 0:
        print("The first line of positive tweets:", positive_tweets[0])
