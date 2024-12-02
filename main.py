import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.fftpack import dct

import scipy.io.wavfile as wavfile
import tempfile
import os

st.set_page_config(page_title="Спектральный анализ голоса", layout="wide")
st.title("Диаризация спикеров")

for key, default_value in {
    'audio_data': None,
    'sample_rate': 16000,
    'fft': None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def record_audio(duration=3):
    recording = sd.rec(int(duration * st.session_state.sample_rate),
                      samplerate=st.session_state.sample_rate,
                      channels=1)
    sd.wait()
    return recording.flatten()

def calculate_mfcc(signal, sample_rate, num_mfcc=13, frame_size=0.025, frame_stride=0.01, num_filters=26, fft_size=512):
    # Step 1: Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    
    # Step 2: Framing
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
    
    # Step 3: Apply Hamming window
    frames *= np.hamming(frame_length)
    
    # Step 4: FFT and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, fft_size))  # Magnitude of FFT
    pow_frames = ((1.0 / fft_size) * (mag_frames ** 2))  # Power Spectrum
    
    # Step 5: Mel Filter Bank
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((fft_size + 1) * hz_points / sample_rate).astype(np.int32)

    fbank = np.zeros((num_filters, int(np.floor(fft_size / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin[m - 1]   # Left
        f_m = bin[m]             # Center
        f_m_plus = bin[m + 1]    # Right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # Step 6: Discrete Cosine Transform (DCT)
    mfcc = dct(filter_banks, axis=1, norm='ortho')[:, :num_mfcc]

    # Step 7: Return MFCC
    return mfcc


def plot_mfcc(mfcc):
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='coolwarm')
    ax.set_title('MFCC Coefficients')
    ax.set_xlabel('Frames')
    ax.set_ylabel('MFCC Coefficients')
    fig.colorbar(cax, ax=ax, orientation='vertical', label='Amplitude (dB)')
    return fig

col1, col2 = st.columns(2)

with col1:
    st.subheader("Управление записью")
    duration = st.slider("Длительность записи (секунды)", 1, 10, 3)

    if st.button("Начать запись"):
        with st.spinner("Запись..."):
            st.session_state.audio_data = record_audio(duration)
        st.success("Запись завершена!")

    if type(st.session_state.audio_data) != None:
        st.audio(st.session_state.audio_data, sample_rate=st.session_state.sample_rate)

    st.divider()

    uploaded_file = st.file_uploader("Загрузить запись", type="wav")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
            
        try:
            sample_rate, audio_data = wavfile.read(tmp_file_path)
            audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
            
            st.session_state.sample_rate = sample_rate
            st.session_state.audio_data = audio_data
        finally:
            os.unlink(tmp_file_path)

    
with col2:
    st.subheader("Расширенное управление")
    sample_rate = st.text_input("Частота дискретизации", value=st.session_state.sample_rate, help="FFF")
    if sample_rate:
        st.session_state.sample_rate = int(sample_rate)

    # st.info("Тут будут всякие кнопочки и формы")


st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["Диаграммы", "Спектрограммы", "Мел-кепстральный анализ", "Показатели обучения"])

with tab1:
    if st.session_state.audio_data is not None:
        st.text(f"Частота дискретизации: {st.session_state.sample_rate}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        time = np.arange(len(st.session_state.audio_data)) / st.session_state.sample_rate
        ax1.plot(time, st.session_state.audio_data)
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Амплитуда (м)')
        ax1.set_title('Форма волны')
        
        fft_data = scipy.fftpack.fft(st.session_state.audio_data)
        freqs = scipy.fftpack.fftfreq(len(fft_data)) * st.session_state.sample_rate

        st.session_state.fft = freqs
        
        pos_mask = freqs >= 0
        ax2.plot(freqs[pos_mask], np.abs(fft_data)[pos_mask])
        ax2.set_xlabel('Частота (Гц)')
        ax2.set_ylabel('Амплитуда')
        ax2.set_title('Частотный спектр')
        ax2.set_xlim(0, 5000)  # Display up to 5kHz
        
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("Запишите аудио, чтобы увидеть анализ!")

with tab2:
    if st.session_state.audio_data is not None:

        fig1, ax1 = plt.subplots(figsize=(12, 4))
        Pxx, freqs, bins, im = ax1.specgram(st.session_state.audio_data, NFFT=1024, noverlap=512, Fs=st.session_state.sample_rate, cmap='viridis')

        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Частота (Гц)')
        ax1.set_title('2D Спектрограмма')
        
        plt.colorbar(im, ax=ax1).set_label('Амплитуда')
        st.pyplot(fig1)

        fig2 = plt.figure(figsize=(20, 12))
        ax2 = fig2.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(bins, freqs)
        surf = ax2.plot_surface(X, Y, 10 * np.log10(Pxx), cmap='viridis')
        
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Частота (Гц)')
        ax2.set_zlabel('Мощность (дБ)')
        ax2.set_title('3D Спектрограмма')
        
        fig2.colorbar(surf, ax=ax2, label='Мощность (дБ)')
        st.pyplot(fig2)
    else:   
        st.info("Спектрограмма будет доступна после добавления записи!")


with tab3:
    if 'audio_data' in st.session_state:
        audio_data = st.session_state.audio_data
        sample_rate = st.session_state.sample_rate

        if type(audio_data) != None:
            signal = audio_data / np.max(np.abs(audio_data))

            # Extract MFCC
            mfcc = calculate_mfcc(signal, sample_rate)
            # mfcc2 = librosa.feature.mfcc(y=audio_data, n_mfcc=13, sr=sample_rate, hop_length=160)


            pl = plot_mfcc(mfcc)
            # pl2 = plot_mfcc(mfcc2)

            st.pyplot(pl)
            # st.pyplot(pl2)