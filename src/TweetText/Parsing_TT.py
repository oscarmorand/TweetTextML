import csv
from tqdm import tqdm
import sys
import Preprocessing_TT as preprocessing
import random
from os.path import exists

#############################################################################
#
# Global parameters
#
#####################

target_idx = 0
feat_start = 1
feat_end = 5
important_feat = 5

target_normalization = {'0': '1', '4': '0'}

raw_dataset_path = '../../datasets/TweetText_Dataset.csv'
clean_dataset_path = '../../datasets/TweetText_Clean_Dataset.csv'
raw_dataset2_path = '../../datasets/SentimentTweets_Dataset.csv'
clean_dataset2_path = '../../datasets/SentimentTweets_Clean_Dataset.csv'

full_header = ["target", "id", "date", "flag", "user", "text"]


def parse_tt(header, data, n_desired):
    if not exists(clean_dataset_path):
        print("Clean dataset not found, parsing raw dataset...")
        parse_raw_tt(header, data)
        print("Parsing complete\n======== CLEANING ========")
        preprocessing.clean_dataset(data)
        print("Cleaning complete, saving the clean dataset...")
        save_clean_tt(clean_dataset_path, header, data)
        print("Saving complete\n")
    else:
        print("Clean dataset found, parsing clean dataset...")
        parse_clean_tt(header, data, n_desired)
        print("Parsing complete\n")


def parse_raw_tt(header, data):

    file1 = open(raw_dataset_path)
    reader = csv.reader(file1, delimiter=',', quotechar='"')

    # No Header Line in the dataset
    header.append(full_header[target_idx])
    header.append(full_header[important_feat])

    # Read data
    length = sum(1 for line in reader)
    file1.seek(0)
    for i in tqdm(range(length), desc="Loading Dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
        row = next(reader)

        # Load Target
        temp = []
        target = target_normalization[row[target_idx]]
        temp.append(target)

        # Load Data
        temp.append(row[important_feat])

        data.append(temp)
    print()

    # Test Print
    print("Full Header [" + str(len(full_header)) + "]:", full_header)
    print("Simplified Header [" + str(len(header)) + "]:", header, "\n")

    print("Data[0]: ", data[0])

    print("str Data[" + str(len(data)) + "]")

    file1.close()


def parse_second_dataset(dataset2_path, data, i_target, i_text):
    with open(dataset2_path, 'r', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)
        length = sum(1 for line in reader)
        csvfile.seek(0)
        next(reader)
        for x in tqdm(range(length), desc="Parsing second dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
            row = next(reader)
            data.append([row[i_target], row[i_text]])
    return length


def merge_sentiment_tweet_dataset(data):
    data2 = []
    print("Let's merge the second dataset in our data...\nLength of data set before merging:", len(data))
    if not exists(clean_dataset2_path):
        print("Second clean dataset not found, parsing second raw dataset...")
        length = parse_second_dataset(raw_dataset2_path, data2, 2, 1)
        print("Parsing complete\nCleaning of second dataset...")
        preprocessing.clean_dataset(data2)
        print("Cleaning complete, saving the second clean dataset...")
        save_clean_tt(clean_dataset2_path, "", data2)
        print("Saving complete")
    else:
        print("Clean dataset found, parsing clean dataset...")
        length = parse_second_dataset(clean_dataset2_path, data2, 0, 1)
        print("Parsing complete")
    l = len(data)
    for x in tqdm(range(length), desc="Merging second dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
        i = random.randint(0, l)
        data.insert(i, data2[x])
        l += 1
    print("Merging complete\nLength of data after merging:", len(data),"\n")


def parse_clean_tt(header, data, n_desired):
    entire_dataset = (n_desired == -1)
    temp = []
    with open(clean_dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for parameter in next(reader):
            header.append(parameter)
        length = sum(1 for line in reader)
        csvfile.seek(0)
        next(reader)
        for i in tqdm(range(length), desc="Loading Clean Dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
            temp.append(next(reader))
    random.shuffle(temp)
    if not entire_dataset:
        for i in range(n_desired):
            data.append(temp[i])
    else:
        for i in range(len(temp)):
            data.append(temp[i])


def save_clean_tt(path, header, data):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header != "":
            writer.writerow(header)
        for row in data:
            writer.writerow(row)