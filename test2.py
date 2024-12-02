import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile





# Load audio file
sample_rate, signal = wavfile.read("samples/sample_test.wav")
signal = signal / np.max(np.abs(signal))  # Normalize signal

# Calculate MFCC
mfcc = calculate_mfcc(signal, sample_rate, num_mfcc=13)

print(mfcc)