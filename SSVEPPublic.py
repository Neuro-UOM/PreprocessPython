import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from IPython import get_ipython
import seaborn as sns
import glob
import pandas as pd
from scipy import signal
#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

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


def fourier(node):
    y = df[node]
    y = butter_highpass_filter(y,5,256,5)   
    ps = np.abs(np.fft.fft(y))**2
    ps = ps[range(y.size / 2) ] 
    '''
    # Scale Data    
    scaler = MinMaxScaler(feature_range=(0, 10000))
    preprocessedPS = scaler.fit_transform(ps)
    '''
    # Standardize Data
    scaler = StandardScaler().fit(ps)
    preprocessedPS = scaler.transform(ps)
    
    time_step = float(1)/256
    '''
    freqs = np.fft.fftfreq( y.size / 2, time_step )
    idx = np.argsort(freqs)
    '''
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n*time_step
    freqs = k/T # two sides frequency range
    freqs = freqs[range(n/2)] # one side frequency range
    idx = np.argsort(freqs)
    
    return freqs,preprocessedPS,idx


file_ = ('./Data/ssvep/SSVEP_8Hz_Trial1_SUBJ1' + ".csv")

df = pd.read_csv(file_)

rows,clmns = df.shape 

# specifying the O2 node for the value

f1,p1,i1 = fourier("A22")
f2,p2,i2 = fourier("A23") 
#f3,p3,i3 = fourier('P7 Value')
#f4,p4,i4 = fourier('P8 Value')
#f5,p5,i5 = fourier('F3 Value')


plt.figure(1)
plt.subplot(211)
#plt.plot(f2[i2] , p2[i2],label="A22")
plt.plot(f1[i1] , p1[i1],label="A23(Oz)")
#plt.plot(f3[i3] , p3[i3],label="P7")
#plt.plot(f4[i4] , p4[i4],label="P8")
#plt.plot(f5[i5] , p5[i5],label="F3")

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)


plt.show()

