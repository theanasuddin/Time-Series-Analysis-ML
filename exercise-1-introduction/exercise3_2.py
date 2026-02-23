#Jari Turunen, TUNI
import numpy as np
from numpy import cos, sin, pi, absolute, arange, mean
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.io import wavfile

fs, data = wavfile.read("Kuusi.wav")
data = data.astype(float)
print(data.shape)

len1 = 4  #length of the average filter (trend) (1+2*len)=window size
len2 = 10  #longer trend (1+2*len2)
x = data.copy() * 0  #fast initialization
x2 = data.copy() * 0
x3 = data.copy() * 0
x4 = data.copy() * 0
energy = data.copy() * 0
for i in range(len(data)):
    print("%d / %d\n" % (i, len(data)))
    start = i - len1
    if start < 1:  #for initializing the window
        start = 1
    start2 = i - len2
    if start2 < 1:  #for initializing the window2
        start2 = 1

    ending = i + len1
    if ending > len(data):  #taking care of the
        ending = len(data)  #end of the window

    ending2 = i + len2
    if ending2 > len(data):  #taking care of the
        ending2 = len(data)  #end of the window2

    if len(data[start:ending]) < 2:
        x[i] = 0
        energy[i] = 0
    else:
        window_data = data[start:ending]
        x[i] = np.mean(data[start:ending])  #sliding window mean
        energy[i] = np.sum((window_data - np.mean(window_data))**2)
    if len(data[start2:ending2]) < 2:
        x2[i] = 0
        x3[i] = 0
        x4[i] = 0
    else:
        x2[i] = np.mean(data[start2:ending2])  #sliding window mean
        x3[i] = skew(data[start2:ending2], axis=0, bias=True)
        x4[i] = kurtosis(data[start2:ending2], axis=0, bias=True)

plt.plot(data)
plt.plot(x, 'r')
plt.plot(x2, 'g')  #plot the results
plt.plot(x3, 'b')
plt.plot(x4, 'y')  #plot the results
plt.plot(energy, 'm')
plt.legend([
    'Original',
    str(len1 * 2 + 1) + '-sample filtered',
    str(len2 * 2 + 1) + '-sample filtered'
])
plt.show()
