import Parsing_TT as parsing
import Preprocessing_TT as preprocessing
from os.path import exists
import random
import matplotlib.pyplot as plt
import numpy as np

header = []
data = []

n_desired = -1

#############################################################################
#
# Exploratory Data Analysis
#
#####################

nb_max_words = 10


def show_most_used_words(words, categorie):
    x = np.arange(nb_max_words)
    width = 0.35

    fig, ax = plt.subplots()
    rects = ax.bar(x - width / 2, [word[1] for word in words], width)

    ax.set_ylabel('Nb Occurrences')
    ax.set_title("Number of occurrences of the "+str(nb_max_words)+"th most used words in "+categorie)
    ax.set_xticks(x, labels=[word[0] for word in words])

    ax.bar_label(rects, padding=3)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':

    parsing.parse_tt(header, data, n_desired)

    print("======== EDA ========")

    targets = [row[0] for row in data]

    negative_tweets = []
    positive_tweets = []
    for line in data:
        if line[0] == '1':
            negative_tweets.append(line[1])
        elif line[0] == '0':
            positive_tweets.append(line[1])

    print("The first line of negative tweets:", negative_tweets[0])
    if len(positive_tweets) > 0:
        print("The first line of positive tweets:", positive_tweets[0])

    categories = {
        "negative tweets": negative_tweets,
        "positive tweets": positive_tweets
    }

    for categorie in categories:
        dict_words = {}
        for line in categories[categorie]:
            temp_words = line.split()
            for word in temp_words:
                if word in dict_words:
                    dict_words[word] += 1
                else:
                    dict_words[word] = 1

        max_words = []
        def take_occ(item):
            return item[1]
        for word in dict_words:
            max_words.append((word, dict_words[word]))
        max_words.sort(key=take_occ, reverse=True)
        max_words = max_words[:nb_max_words]

        print(str(nb_max_words)+"th most used words in", categorie+":", max_words)
        show_most_used_words(max_words, categorie)
