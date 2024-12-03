from itertools import islice
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io.wavfile as wavfile

from datasets import load_dataset

from model.classification_model import SpeakerClassifier

from utils.rtcvad_helpers import split_audio

ds = load_dataset("edinburghcstr/ami", "ihm") # , trust_remote_code=True

records_paths = [ p.get("path")  for p in ds.data['train'].to_pandas()['audio'] ]

class AudioDataset(Dataset):
    def __init__(self, file_paths, max_sequence_length=500):
        # Filter out invalid files during initialization
        self.file_paths = []
        for path in file_paths:
            try:
                sample_rate, audio_data = wavfile.read(path)
                if len(audio_data) > 0:  # Check if audio data is not empty
                    self.file_paths.append(path)
            except Exception as e:
                print(f"Skipping invalid audio file {path}: {str(e)}")
                
        if not self.file_paths:
            raise ValueError("No valid audio files found in the dataset")
            
        self.max_sequence_length = max_sequence_length
    
    def __len__(self):
        return len(self.file_paths)
    
    def pad_or_truncate(self, features):
        if features.shape[0] > self.max_sequence_length:
            return features[:self.max_sequence_length, :]
        elif features.shape[0] < self.max_sequence_length:
            padding_length = self.max_sequence_length - features.shape[0]
            padding = np.zeros((padding_length, features.shape[1]))
            return np.vstack((features, padding))
        return features
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        try:
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Check for valid audio data
            if len(audio_data) == 0:
                raise ValueError(f"Empty audio file: {audio_path}")
                
            signal = audio_data.astype(np.float32)
            
            # Normalize using a safe method
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
            else:
                signal = signal / np.iinfo(np.int16).max
            
            # Calculate features
            features = SpeakerClassifier.calculate_mfcc(signal, sample_rate)
            
            # Ensure we have valid features
            if features is None or features.size == 0:
                raise ValueError(f"Failed to extract features from {audio_path}")
            
            # Pad or truncate to fixed length
            features = self.pad_or_truncate(features)
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing file {audio_path}: {str(e)}")
            # Return a zero tensor of the correct shape as a fallback
            return torch.zeros((self.max_sequence_length, 39), dtype=torch.float32)

def train():
    # Hyperparameters
    batch_size = 32
    num_epochs = 1000
    learning_rate = 0.001
    patience = 10
    max_sequence_length = 500  # Define maximum sequence length
    
    # Split dataset
    train_paths, val_paths = train_test_split(records_paths[:5000], test_size=0.2, random_state=42)
    
    # Create data loaders with fixed sequence length
    train_dataset = AudioDataset(train_paths, max_sequence_length=max_sequence_length)
    val_dataset = AudioDataset(val_paths, max_sequence_length=max_sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    autoencoder = SpeakerClassifier(39, 16)  # 39 features (13 MFCC + 13 delta + 13 delta2)
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        # Training phase
        autoencoder.train()
        train_loss = 0
        for batch in train_loader:
            # Reshape batch for processing
            batch = batch.view(-1, 39)  # Flatten the sequence dimension
            
            optimizer.zero_grad()
            _, reconstructed = autoencoder(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.view(-1, 39)  # Flatten the sequence dimension
                _, reconstructed = autoencoder(batch)
                val_loss += criterion(reconstructed, batch).item()
        
        # Calculate average losses
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     best_epoch = epoch
        #     patience_counter = 0
        #     autoencoder.save_weights()
        #     print("Model saved! New best validation loss achieved.")
        # else:
        #     patience_counter += 1
        #     print(f"Patience counter: {patience_counter}/{patience}")
            
        #     if patience_counter >= patience:
        #         print(f"Early stopping triggered! Best epoch was {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
        #         # Load the best model weights
        #         autoencoder.load_weights()
        #         break
    
    print("Training completed!")
    print(f"Best model was from epoch {best_epoch + 1} with validation loss: {best_val_loss:.4f}")
    
    return autoencoder
    


    

