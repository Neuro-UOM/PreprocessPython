# -*- coding: utf-8 -*-
"""
Created on Thu Aug 03 14:48:11 2017
"""
from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.dates as dates

_file = "AllRaw.csv"
df = pd.read_csv(_file,index_col=None, header=0)

x1 = df['Timestamp']
new_x = dates.datestr2num(x1)
y1 = df['O1 Value']
y2 = df['O2 Value']
y3 = df['P8 Value']
y4 = df['P7 Value']
y5 = df['T8 Value']
y6 = df['T7 Value']

plt.plot(x1, y1)