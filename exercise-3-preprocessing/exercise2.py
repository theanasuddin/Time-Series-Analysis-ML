# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import noisereduce as nr

# two speech samples
file1 = "AirportAnnouncements_1.wav"
file2 = "AirportAnnouncements_2.wav"

def process_audio(filepath):
    Fs, data = wav.read(filepath)
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data[:, 0]
    data = data / np.max(np.abs(data))
    reduced = nr.reduce_noise(y=data, sr=Fs, y_noise=data[0:1000])
    return Fs, data, reduced

Fs1, data1, reduced1 = process_audio(file1)
Fs2, data2, reduced2 = process_audio(file2)

# clip 300-500 samples
clip1_orig = data1[300:500]
clip1_red = reduced1[300:500]
clip2_orig = data2[300:500]
clip2_red = reduced2[300:500]

def plot_spectrum(signal, Fs, title):
    plt.specgram(signal, NFFT=64, Fs=Fs, noverlap=32, mode='psd', scale='dB')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Power (dB)")

# create figure with 4 plots
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plot_spectrum(clip1_orig, Fs1, "AirportAnnouncements_1 Original (300–500)")

plt.subplot(2, 2, 2)
plot_spectrum(clip1_red, Fs1, "AirportAnnouncements_1 Reduced (300–500)")

plt.subplot(2, 2, 3)
plot_spectrum(clip2_orig, Fs2, "AirportAnnouncements_2 Original (300–500)")

plt.subplot(2, 2, 4)
plot_spectrum(clip2_red, Fs2, "AirportAnnouncements_2 Reduced (300–500)")

plt.tight_layout()
plt.show()
