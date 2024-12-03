from types import NoneType
import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.fftpack
import matplotlib.pyplot as plt

import scipy.io.wavfile as wavfile
import tempfile
import os

from model.classification_model import SpeakerClassifier, inference
from utils.chart_helpers import chart_labels

st.set_page_config(page_title="Спектральный анализ голоса", layout="wide")
st.title("Диаризация спикеров")

for key, default_value in {
    'audio_data': None,
    'num_speakers': 2,
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
    num_speakers = st.text_input("Колличество классов спикеров", value=st.session_state.num_speakers, help="FFF")
    
    if sample_rate:
        st.session_state.sample_rate = int(sample_rate)
    if  num_speakers:
        st.session_state.num_speakers = int(num_speakers)

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

        if type(audio_data) != NoneType:
            signal = audio_data / np.max(np.abs(audio_data))

            mfcc = SpeakerClassifier.calculate_mfcc(signal, sample_rate)
            pl = plot_mfcc(mfcc)

            st.pyplot(pl)
with tab4:
    if 'audio_data' in st.session_state:
        audio_data = st.session_state.audio_data
        sample_rate = st.session_state.sample_rate
        num_speakers = st.session_state.num_speakers

        if type(audio_data) != NoneType:
            # signal = audio_data / np.max(np.abs(audio_data))

            labels = inference(audio_data, sample_rate, num_speakers)
            
            plt = chart_labels(labels, len(signal)/sample_rate, num_speakers)

            st.pyplot(plt)