import Parsing_TT as parsing
import Preprocessing_TT as preprocessing
from os.path import exists

simplified_header = []
data = []
target = []
data_with_target = []

#############################################################################
#
# Exploratory Data Analysis
#
#####################

if __name__ == '__main__':

    if not exists("../../../datasets/TweetText_Clean_Dataset.csv"):
        print("Clean dataset not found, parsing raw dataset...")
        parsing.parse_raw_tt(simplified_header, data, target, data_with_target)
        print("Parsing complete\n======== CLEANING ========")
        preprocessing.clean_dataset(data)
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
