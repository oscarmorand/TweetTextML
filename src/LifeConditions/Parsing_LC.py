import csv

#############################################################################
#
# Global parameters
#
#####################

target_idx=22
feat_start=0
feat_end=21


def parse_lc(header, data, target, data_with_target):

    file1 = csv.reader(open('../../datasets/LifeConditions_Dataset.csv'), delimiter=',', quotechar='"')

    # Read Header Line
    for parameter_name in next(file1):
        header.append(parameter_name)

    # Read data
    for row in file1:
        # Load Data
        temp = []
        for j in range(feat_start, feat_end + 1):
            if row[j] == '':
                temp.append(float())
            else:
                temp.append(float(row[j]))
        data.append(temp)

        # Load Target
        if row[target_idx] == '':
            continue
        else:
            target.append(float(row[target_idx]))
            temp.insert(0, float(row[target_idx]))  # Add the target at the beginning of the data

        data_with_target.append(temp)  # Load everything in the complete array

    # Test Print
    print("Header [" + str(len(header)) + "]:")
    print(header)
    print("Data [" + str(len(data)) + "," + str(len(data[0])) + "]")
    print(str(len(target)) + " targets, " + str(len(data)) + " data", end=': ')
    if len(target) == len(data):
        print("same lengths, OK")
    else:
        print("different lengths, NOT OK")