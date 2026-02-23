import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

Fs, y = wavfile.read('./Kuusi.wav')
y = y.astype(np.float64)
y = y - np.mean(y)
y = y / np.max(np.abs(y))

time = np.arange(len(y)) / Fs
plt.figure(figsize=(12, 4))
plt.plot(time, y)
plt.title('Speech Time Series (Normalized)')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
