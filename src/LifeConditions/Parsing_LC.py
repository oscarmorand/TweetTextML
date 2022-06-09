import csv

#############################################################################
#
# Global parameters
#
#####################

target_idx = 22
feat_start = 0
feat_end = 21

dataset_path = '../../datasets/LifeConditions_Dataset.csv'


def parse_lc(header, data):
    with open(dataset_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')

        # Read Header Line
        for parameter_name in next(reader):
            header.append(parameter_name)

        # Read data
        for row in reader:

            temp = []
            # Load Target
            if row[target_idx] == '':
                continue
            else:
                temp.append(row[target_idx])

            # Load Data
            temp2 = []
            for j in range(feat_start, feat_end + 1):
                if row[j] == '':
                    temp2.append(float())
                else:
                    temp2.append(float(row[j]))
            temp.append(temp2)

            data.append(temp)

        # Test Print
        print("Header [" + str(len(header)) + "]:")
        print(header)
        print("Data [" + str(len(data)) + "," + str(len(data[0][1])) + "]")