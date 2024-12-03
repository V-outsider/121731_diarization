from model.classification_model import inference
import scipy.io.wavfile as wavfile

import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    sample_rate, audio_data = wavfile.read("samples/check_duo_10s.wav")
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    
    labels = inference(audio_data, sample_rate, 2)
    
    duration = len(labels) * 10
    
    time_axis = np.linspace(0, duration, len(labels))
    
    plt.figure(figsize=(10, 4))
    
    plt.plot(time_axis, labels, drawstyle='steps-pre', label="Speaker Cluster")
    plt.xlabel("Time (s)")
    plt.ylabel("Cluster Label")
    plt.title("Speaker Activity Over Time")
    plt.legend()
    plt.show()