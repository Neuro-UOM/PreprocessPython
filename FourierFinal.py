# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from IPython import get_ipython
import seaborn as sns
import glob
import pandas as pd
from scipy import signal

# extracted from : https://stackoverflow.com/questions/39032325/python-high-pass-filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

#file_ = ('./Data/VEP/10hz/1sec' + ".csv")
#file_ = ('./Data/VEP/RAW/raw_hansika_red' + ".csv")
file_ = ('./Data/ssvep/nadun_11' + ".csv")

df = pd.read_csv(file_,index_col=None, header=0)

# specifying the O2 node for the value
y = df['O1 Value']
y = butter_highpass_filter(y,5,132,5)

ps = np.abs(np.fft.fft(y))**2

print np.mean(ps)
print ps[0]
time_step = float(1)/128
freqs = np.fft.fftfreq( y.size , time_step )
idx = np.argsort(freqs)

plt.plot(freqs[idx] , ps[idx])

plt.show()

