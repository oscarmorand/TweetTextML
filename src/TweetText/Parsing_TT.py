import csv
from tqdm import tqdm
import sys

#############################################################################
#
# Global parameters
#
#####################

target_idx = 0
feat_start = 1
feat_end = 5
important_feat = 5
entire_dataset = False
n_desired = 1000



def parse_raw_tt(simplified_header, data, target, all_data):

    file1 = open('../../../datasets/TweetText_Dataset.csv')
    reader = csv.reader(file1, delimiter=',', quotechar='"')

    # No Header Line in the dataset
    header = ["target", "id", "date", "flag", "user", "text"]
    simplified_header.append("target")
    simplified_header.append("text")

    # Read data
    length = sum(1 for line in reader)
    file1.seek(0)
    for i in tqdm(range(length), desc="Loading Dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
        row = next(reader)
        if i > n_desired - 1:
            if not entire_dataset:
                break

        # Load Target
        temp = []
        target.append(row[target_idx])  # Add the target on the target array
        temp.append(row[target_idx])

        # Load Data
        temp.append(row[important_feat])
        data.append(temp)
        full_temp = []
        for x in row:
            full_temp.append(x)
        all_data.append(full_temp)
    print()

    # Test Print
    print("Full Header [" + str(len(header)) + "]:", header)
    print("Simplified Header [" + str(len(simplified_header)) + "]:", simplified_header, "\n")

    print("Data[0]: ", data[0])
    print("Target[0]: ", target[0])
    print("AllData[0]: ", all_data[0], '\n')

    print("str Data[" + str(len(data)) + "]")
    print(str(len(target)) + " targets, " + str(len(data)) + " data", end=': ')
    if len(target) == len(data):
        print("same lengths, OK")
    else:
        print("different lengths, NOT OK")

    file1.close()


def parse_clean_tt(header, data):
    with open('../../../datasets/TweetText_Clean_Dataset.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for parameter in next(reader):
            header.append(parameter)
        length = sum(1 for line in reader)
        if not entire_dataset:
            pos_n, neg_n = n_desired/2, n_desired/2
        csvfile.seek(0)
        next(reader)
        for i in tqdm(range(length), desc="Loading Clean Dataset...", ascii=False, ncols=100, position=0, leave=True, file=sys.stdout):
            row = next(reader)
            if not entire_dataset:
                if (row[0] == '0') and (neg_n > 0):
                    neg_n -= 1
                elif (row[0] == '4') and (pos_n > 0):
                    pos_n -= 1
                else:
                    continue
            data.append(row)


def save_clean_tt(header, data):
    with open('../../../datasets/TweetText_Clean_Dataset.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for row in data:
            writer.writerow(row)