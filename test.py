import librosa
import numpy as np

import scipy.io.wavfile as wavfile

sample_rate, audio_data = wavfile.read("samples/sample_test.wav")
audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            

hop_length = 160 # number of samples between successive frames
mfcc = librosa.feature.mfcc(y=audio_data, n_mfcc=13, sr=sample_rate, hop_length=hop_length)

audio_length = len(audio_data) / sample_rate # in seconds
step = hop_length / sample_rate # in seconds
intervals_s = np.arange(start=0, stop=audio_length, step=step)

print(f'MFCC shape: {mfcc.shape}')
print(f'intervals_s shape: {intervals_s.shape}')
print(f'First 5 intervals: {intervals_s[:5]} second')