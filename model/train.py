from itertools import islice
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io.wavfile as wavfile

from datasets import load_dataset

from model.classification_model import SpeakerClassifier

from utils.rtcvad_helpers import split_audio

ds = load_dataset("edinburghcstr/ami", "ihm") # , trust_remote_code=True

records_paths = [ p.get("path")  for p in ds.data['train'].to_pandas()['audio'] ]

def train():
    batch_size = 5
    batch_count = 10
    num_epochs = 1000

    autoencoder = SpeakerClassifier(13, 16)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        autoencoder.train()
        epoch_loss = 0

        # Randomly sample batches from the streaming dataset
        for _ in range(batch_count):  # Limit to 100 batches per epoch
            batch = list(islice(records_paths, batch_size))

            # Extract MFCC features for the batch
            batch_features = []
            for audio in batch:
                sample_rate, audio_data = wavfile.read(audio)
                signal = audio_data.astype(np.float32) / np.iinfo(np.int16).max

                mfcc = SpeakerClassifier.calculate_mfcc(signal, sample_rate, num_mfcc=13)
                batch_features.append(mfcc)

            # Flatten and convert to tensor
            batch_features = torch.tensor(np.vstack(batch_features), dtype=torch.float32)

            # Train autoencoder
            _, reconstructed = autoencoder(batch_features)
            loss = criterion(reconstructed, batch_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    

    autoencoder.save_weights()
    
    print("[INFO] - Weights was saved")
    


    

