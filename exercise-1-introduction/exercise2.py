import librosa
import numpy as np
import pygame
import scipy.io.wavfile as wavfile
import tempfile
import os
import matplotlib.pyplot as plt

# Test different sampling rates to find lowest understandable rate
sampling_rates = [8000, 4000, 2000, 1000, 500]
responses = {}

plt.figure(figsize=(12, 8))

for i, sr in enumerate(sampling_rates, start=1):
    print(f"\nTesting sampling rate: {sr} Hz")

    y_low, s = librosa.load("Kuusi.wav", sr=sr)

    audio = y_low * (2**15 - 1) / np.max(np.abs(y_low))
    audio = audio.astype(np.int16)

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(temp_file.name, sr, audio)
    temp_file.close()

    pygame.mixer.init(frequency=sr)
    pygame.mixer.music.load(temp_file.name)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)

    response = input("Can you understand the message? (y/n): ")
    responses[sr] = response.lower()

    pygame.mixer.quit()
    os.unlink(temp_file.name)

    plt.subplot(len(sampling_rates), 1, i)
    time = np.arange(len(y_low)) / sr
    plt.plot(time, y_low)
    plt.title(f"Speech Signal Downsampled to {sr} Hz")
    plt.xlabel('Time (seconds)')
    plt.ylabel("Amplitude")

    if response.lower() == "n":
        print(f"Message becomes unclear at {sr} Hz")
        break

plt.tight_layout()
plt.savefig("downsampling_results.png")
plt.show()

print("\nUser responses:", responses)
