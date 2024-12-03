import os
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim

from scipy.fftpack import dct

class SpeakerClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(SpeakerClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

    @classmethod
    def calculate_mfcc(cls, signal, sample_rate, num_mfcc=13, frame_size=0.025, frame_stride=0.01, num_filters=26, fft_size=512):
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # Разбиение на фреймы
        frame_length = int(frame_size * sample_rate)
        frame_step = int(frame_stride * sample_rate)
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1


        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        padded_signal = np.append(emphasized_signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = padded_signal[indices.astype(np.int32, copy=False)]
        
        # Окно Хэмминга
        frames *= np.hamming(frame_length)
        
        # FFT и Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, fft_size))  # Магнитуда FFT
        pow_frames = ((1.0 / fft_size) * (mag_frames ** 2))  # Power Spectrum
        
        # Набор Мел-Фильтров
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Конвертация из Герц в Мел
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Из Мел в Герц
        bin = np.floor((fft_size + 1) * hz_points / sample_rate).astype(np.int32)

        fbank = np.zeros((num_filters, int(np.floor(fft_size / 2 + 1))))
        for m in range(1, num_filters + 1):
            f_m_minus = bin[m - 1]   # Лево
            f_m = bin[m]             # Центр
            f_m_plus = bin[m + 1]    # Право

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Числовая стабильность
        filter_banks = 20 * np.log10(filter_banks)  # В децибелах

        # Дискретное косинусное преобразование (DCT)
        mfcc = dct(filter_banks, axis=1, norm='ortho')[:, :num_mfcc]

        return mfcc
    
    def save_weights(self):
        torch.save(self.state_dict(), 'model/weights/model_weights.pth')

    def load_weights(self):
        self.load_state_dict(torch.load("model/weights/model_weights.pth"))

    @classmethod
    def cluster_speakers(cls, latent_features, num_speakers):
        kmeans = KMeans(n_clusters=num_speakers, random_state=42)
        labels = kmeans.fit_predict(latent_features)
        return labels
    
def inference(signal, sample_rate=16000, num_speakers=2):
    autoencoder = SpeakerClassifier(13, 16)

    autoencoder.eval()

    if os.path.exists("model/weights/model_weights.pth"):
        autoencoder.load_weights()
        print("[INFO] - Weights was loaded")
    else:
        raise NotImplementedError("Не существует готовых к использованию весов")
    
    all_latent_features = []
    batch_features = []
    
    mfcc = SpeakerClassifier.calculate_mfcc(signal, sample_rate, num_mfcc=13)
    batch_features.append(mfcc)
    
    batch_features = torch.tensor(np.vstack(batch_features), dtype=torch.float32)
    
    with torch.no_grad():
        latent, _ = autoencoder(batch_features)

    all_latent_features.append(latent.numpy())
    all_latent_features = np.vstack(all_latent_features)

    num_speakers = 3
    labels = SpeakerClassifier.cluster_speakers(all_latent_features, num_speakers)

    print(f"Predicted number of speakers: {num_speakers}")
    print(f"Cluster labels: {labels}")
    
    return labels
