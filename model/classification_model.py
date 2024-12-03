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
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return embedding, reconstruction

    @classmethod
    def calculate_mfcc(cls, signal, sample_rate, num_mfcc=13, frame_size=0.025, frame_stride=0.01):
        # Check for valid input
        if signal is None or len(signal) == 0:
            raise ValueError("Empty or invalid signal provided")
        
        # Ensure signal is not all zeros
        if np.all(signal == 0):
            raise ValueError("Signal contains only zeros")
        
        # Safe normalization
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
        
        # Calculate basic MFCC
        try:
            mfcc = cls._calculate_base_mfcc(signal, sample_rate, num_mfcc, frame_size, frame_stride)
            
            # Check if MFCC calculation produced valid results
            if mfcc is None or mfcc.size == 0:
                raise ValueError("MFCC calculation produced empty results")
            
            # Calculate delta features
            delta = cls._calculate_delta(mfcc)
            delta2 = cls._calculate_delta(delta)
            
            # Combine features
            features = np.concatenate([mfcc, delta, delta2], axis=1)
            
            # Apply mean and variance normalization
            features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
            
            return features
            
        except Exception as e:
            raise ValueError(f"Error in MFCC calculation: {str(e)}")

    @staticmethod
    def _calculate_delta(features, N=2):
        padded = np.pad(features, ((N, N), (0, 0)), mode='edge')
        delta = np.zeros_like(features)
        
        for t in range(len(features)):
            delta[t] = np.dot(np.arange(-N, N+1),
                            padded[t:t+2*N+1]) / (2 * sum([i**2 for i in range(1, N+1)]))
        return delta
    
    def save_weights(self):
        torch.save(self.state_dict(), 'model/weights/model_weights.pth')

    def load_weights(self):
        self.load_state_dict(torch.load("model/weights/model_weights.pth"))

    @classmethod
    def cluster_speakers(cls, latent_features, num_speakers):
        kmeans = KMeans(n_clusters=num_speakers, random_state=42)
        labels = kmeans.fit_predict(latent_features)
        return labels
    
    @staticmethod
    def _calculate_base_mfcc(signal, sample_rate, num_mfcc=13, frame_size=0.025, frame_stride=0.01, num_filters=26, fft_size=512):
        # Add minimum signal length check
        min_samples = int(frame_size * sample_rate)
        if len(signal) < min_samples:
            raise ValueError(f"Signal too short. Minimum length: {min_samples} samples")
        
        # Pre-emphasis
        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        
        # Frame splitting
        frame_length = int(frame_size * sample_rate)
        frame_step = int(frame_stride * sample_rate)
        signal_length = len(emphasized_signal)
        num_frames = int(np.ceil((signal_length - frame_length) / frame_step)) + 1

        # Zero padding
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        padded_signal = np.append(emphasized_signal, z)

        # Split into frames
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
                np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = padded_signal[indices.astype(np.int32, copy=False)]
        
        # Apply Hamming window
        frames *= np.hamming(frame_length)
        
        # FFT and Power Spectrum
        mag_frames = np.absolute(np.fft.rfft(frames, fft_size))
        pow_frames = ((1.0 / fft_size) * (mag_frames ** 2))
        
        # Mel filterbank
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)
        hz_points = (700 * (10**(mel_points / 2595) - 1))
        bin = np.floor((fft_size + 1) * hz_points / sample_rate).astype(np.int32)

        fbank = np.zeros((num_filters, int(np.floor(fft_size / 2 + 1))))
        for m in range(1, num_filters + 1):
            f_m_minus = bin[m - 1]
            f_m = bin[m]
            f_m_plus = bin[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        # Apply mel filterbank
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        # DCT
        mfcc = dct(filter_banks, axis=1, norm='ortho')[:, :num_mfcc]
        
        # Mean normalization
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
        
        return mfcc

def inference(signal, sample_rate=16000, num_speakers=2, window_size=4.0):
    autoencoder = SpeakerClassifier(39, 16)  # 39 = 13 MFCC + 13 delta + 13 delta2

    if not os.path.exists("model/weights/model_weights.pth"):
        raise NotImplementedError("Не существует готовых к использованию весов")
    
    autoencoder.load_weights()
    autoencoder.eval()

    # Split long audio into windows
    window_samples = int(window_size * sample_rate)
    stride_samples = window_samples // 2  # 50% overlap
    
    all_embeddings = []
    
    # Process audio in windows
    for start in range(0, len(signal), stride_samples):
        end = start + window_samples
        if end > len(signal):
            break
            
        window = signal[start:end]
        
        # Calculate features
        mfcc_features = SpeakerClassifier.calculate_mfcc(window, sample_rate)
        
        # Convert to tensor
        features_tensor = torch.tensor(mfcc_features, dtype=torch.float32)
        
        # Get embeddings
        with torch.no_grad():
            embedding, _ = autoencoder(features_tensor)
            all_embeddings.append(embedding.numpy())
    
    all_embeddings = np.vstack(all_embeddings)
    
    # Perform clustering
    labels, confidences = cluster_speakers_with_confidence(all_embeddings, num_speakers)
    
    # Apply temporal smoothing
    smoothed_labels = temporal_smoothing(labels)
    
    return smoothed_labels, confidences

def cluster_speakers_with_confidence(embeddings, num_speakers):
    kmeans = KMeans(n_clusters=num_speakers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Calculate confidence scores
    distances = kmeans.transform(embeddings)
    confidences = 1 / (1 + distances)
    
    return labels, confidences

def temporal_smoothing(labels, window_size=5):
    smoothed = np.copy(labels)
    pad_width = window_size // 2
    padded = np.pad(labels, (pad_width, pad_width), mode='edge')
    
    for i in range(len(labels)):
        window = padded[i:i + window_size]
        smoothed[i] = np.bincount(window).argmax()
    
    return smoothed
