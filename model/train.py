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
# ds = load_dataset("edinburghcstr/ami", "ihm", split="train", streaming=True)

def main():
    batch_size = 5
    batch_count = 10
    num_epochs = 1000
    latent_size = 16

    autoencoder = SpeakerClassifier(13, latent_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    if os.path.exists("model/weights/model_weights.pth"):
        autoencoder.load_weights()
        print("[INFO] - Weights was loaded")

    else:
        #  Train Autoencoder on Streaming Batches
        for epoch in range(num_epochs):
            autoencoder.train()
            epoch_loss = 0

            # Randomly sample batches from the streaming dataset
            for _ in range(batch_count):  # Limit to 100 batches per epoch
                batch = list(islice(records_paths, batch_size))

                # Extract MFCC features for the batch
                batch_features = []
                for audio in batch:
                    # signal = np.array(audio["audio"]["array"])
                    # sr = audio["audio"]["sampling_rate"]

                    sample_rate, audio_data = wavfile.read(audio)
                    signal = audio_data.astype(np.float32) / np.iinfo(np.int16).max

                    mfcc = SpeakerClassifier.calculate_mfcc(signal, sample_rate, num_mfcc=13)
                    batch_features.append(mfcc)

                # Flatten and convert to tensor
                batch_features = torch.tensor(np.vstack(batch_features), dtype=torch.float32)

                # Train autoencoder
                latent, reconstructed = autoencoder(batch_features)
                loss = criterion(reconstructed, batch_features)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        

        autoencoder.save_weights()
        
        print("[INFO] - Weights was saved")
    

    autoencoder.eval()
    segments = split_audio("samples/check_duo_10s.wav")
    
    all_latent_features = []
    batch_features = []

    for segment in segments:
        sample_rate, audio_data = wavfile.read(segment.get("path"))
        signal = audio_data.astype(np.float32) / np.iinfo(np.int16).max
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
    # Step 5: Extract Latent Features
    # autoencoder.eval()
    # all_latent_features = []
    # for _ in range(100):  # Process 100 batches for clustering
    #     batch = list(islice(records_paths, batch_size))
    #     batch_features = []
    #     for audio in batch:
    #         sample_rate, audio_data = wavfile.read(audio)
    #         signal = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    #         mfcc = SpeakerClassifier.calculate_mfcc(signal, sample_rate, num_mfcc=13)
    #         batch_features.append(mfcc)

    #     batch_features = torch.tensor(np.vstack(batch_features), dtype=torch.float32)
    #     with torch.no_grad():
    #         latent, _ = autoencoder(batch_features)
    #     all_latent_features.append(latent.numpy())

    # # Combine all latent features
    # all_latent_features = np.vstack(all_latent_features)

    # # Step 6: Perform Clustering
    # num_speakers = 12  # Adjust as necessary
    # labels = SpeakerClassifier.cluster_speakers(all_latent_features, num_speakers)

    # # Step 7: Output Results
    # print(f"Predicted number of speakers: {num_speakers}")
    # print(f"Cluster labels: {labels}")

