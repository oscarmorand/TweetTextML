import csv

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


def parse_tt(simplified_header, data, target, all_data):

    file1 = csv.reader(open('../../datasets/TweetText_Dataset.csv'), delimiter=',', quotechar='"')

    # No Header Line in the dataset
    header = ["target", "id", "date", "flag", "user", "text"]
    simplified_header.append("target")
    simplified_header.append("text")

    # Read data
    i = 0
    for row in file1:
        if i > n_desired - 1:
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

        i += 1

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
    print('\n')