# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import pandas as pd

def createCSV(path1,name):
    path = path1

    allFiles = glob.glob(path + "/*.csv")
    frame = pd.DataFrame()
    list_ = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        value = find_between_r(file_,"_",".csv")
        df.insert(32,"label",value)
        list_.append(df)
    frame = pd.concat(list_)
    
    
    features = []
    waves = ["Low_beta","High_beta","Alpha","Theta", "Gamma"]
    for i in range(7,13):
        for j in waves:
            features.append(str(i)+ " "+ j)
    
    features.append("label")
    print features
    
    frame = frame[features]
#    frame['Label'] = frame['Label'].map({'null': 2, 'green': 1, 'red': 0})
    
    frame.to_csv(name)
    
    
# https://stackoverflow.com/questions/3368969/find-string-between-two-substrings
def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""
        
createCSV('data/VEP/SDK','AllSDK.csv')
