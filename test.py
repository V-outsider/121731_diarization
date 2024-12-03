from model.classification_model import inference
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import numpy as np

def visualize_diarization(signal, sample_rate, labels, confidences):
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), height_ratios=[2, 2, 1])
    
    # Plot waveform
    time = np.linspace(0, len(signal)/sample_rate, len(signal))
    ax1.plot(time, signal)
    ax1.set_title('Waveform')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Plot speaker labels
    duration = len(signal)/sample_rate
    time_labels = np.linspace(0, duration, len(labels))
    ax2.plot(time_labels, labels, drawstyle='steps-post')
    ax2.set_title('Speaker Diarization')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Speaker ID')
    ax2.set_ylim(-0.5, max(labels) + 0.5)
    ax2.grid(True)
    
    # Plot confidence scores
    im = ax3.imshow(confidences.T, aspect='auto', origin='lower', 
                    extent=[0, duration, 0, confidences.shape[1]-1])
    ax3.set_title('Confidence Scores')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speaker ID')
    plt.colorbar(im, ax=ax3, label='Confidence')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # Load audio file
    audio_path = "samples/check_duo_10s.wav"
    try:
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        else:
            audio_data = audio_data / np.iinfo(np.int16).max
        
        # Set number of speakers
        num_speakers = 2
        
        # Get diarization results
        print("Performing speaker diarization...")
        labels, confidences = inference(audio_data, sample_rate, num_speakers)
        
        # Visualize results
        print("Creating visualization...")
        fig = visualize_diarization(audio_data, sample_rate, labels, confidences)
        
        # Save or show the plot
        plt.savefig('diarization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Analysis complete!")
        print(f"Number of speakers detected: {num_speakers}")
        print(f"Total duration: {len(audio_data)/sample_rate:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing audio file: {str(e)}")