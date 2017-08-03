# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing

def createRawCSV(path1,name):
    path = path1
    
    # Get all the csvs in the path directory
    allFiles = glob.glob(path + "/*.csv")
    #empty data frame
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        # Read csv to a dataframe
        df = pd.read_csv(file_,index_col=None, header=0)
        # File the lable from raw_bibile_blue 
        value = find_between_r(file_,"_",".csv")
        # Insert the label to the dataframe 
        df.insert(32,"label",value)
        # Append dataframe to the global list
        list_.append(df)
        
    # create a dataframe from the list
    frame = pd.concat(list_)
#    features = []
    
#    features.append("Label")
#    print features
    
#    frame = frame[features]
#    frame['Label'] = frame['Label'].map({'null': 2, 'green': 1, 'red': 0})
    # Dropping quality columns and gyroscope columns
    frame.drop(frame.columns[[2,4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,31]], axis=1, inplace=True)
    # Write to the file
    frame.to_csv(name)


# https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""

createRawCSV('./Data/VEP/RAW','AllRaw.csv')
