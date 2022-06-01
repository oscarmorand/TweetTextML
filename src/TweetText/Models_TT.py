import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import nltk
from threading import Lock, Thread
import multiprocessing as mp
import time
from tqdm import tqdm
from os.path import exists
import sys
import Parsing_TT as parsing
import Preprocessing_TT as preprocessing


simplified_header = []
data = []
target = []
data_with_target = []

#############################################################################
#
# Model Building
#
#####################

if __name__ == '__main__':

    if not exists("../../datasets/TweetText_Clean_Dataset.csv"):
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

    print("======== Model Building ========")


