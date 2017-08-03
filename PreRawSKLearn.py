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

#x1 = df['Unnamed: 0']
y1 = df['O1 Value']
y2 = df['O2 Value']
y3 = df['P8 Value']
y4 = df['P7 Value']
y5 = df['T8 Value']
y6 = df['T7 Value']

#plt.plot(x1, y1, 'r', x1, y2, 'b', x1, y3, 'g',x1, y4, 'y', x1, y5, 'c', x1, y6, 'm')

redDF = df.loc[df['label'] == 'red']
greenDF = df.loc[df['label'] == 'green']
blueDF = df.loc[df['label'] == 'blue']


yRed = redDF['O1 Value']
xRed = redDF['Unnamed: 0']
yGreen = greenDF['O1 Value']
xGreen = greenDF['Unnamed: 0']
yBlue = blueDF['O1 Value']
xBlue = blueDF['Unnamed: 0']

plt.plot(xRed, yRed, 'r', xGreen, yGreen, 'g', xBlue, yBlue, 'b')