import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import scipy.signal as signal
import soundfile as sf
import tempfile
import librosa
import librosa.display
import io
import os

st.title("🎚️ Digital Music Equalizer")
st.markdown("Adjust bass, midrange, and treble using FIR bandpass filters.")

# Upload audio
uploaded_file = st.sidebar.file_uploader("Upload Audio File (WAV/MP3)", type=["wav", "mp3"])

# Sliders for gain adjustment
st.sidebar.header("Equalizer Settings (Gain in dB)")
bass_gain = st.sidebar.slider("Bass (60–250 Hz)", -20, 20, 0)
mid_gain = st.sidebar.slider("Midrange (250–4000 Hz)", -20, 20, 0)
treble_gain = st.sidebar.slider("Treble (4k–10k Hz)", -20, 20, 0)

fs_target = 44100  # Standard audio sampling rate

@st.cache_data
def fir_bandpass(lowcut, highcut, fs, numtaps=101):
    nyq = fs / 2
    taps = signal.firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
    return taps

@st.cache_data
def apply_filter(audio, taps):
    return signal.lfilter(taps, 1.0, audio)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        filepath = tmp.name

    y, sr = librosa.load(filepath, sr=fs_target, mono=True)
    duration = len(y) / sr
    t = np.linspace(0, duration, len(y))

    # Design FIR filters
    bass_filt = fir_bandpass(60, 250, sr)
    mid_filt = fir_bandpass(250, 4000, sr)
    treble_filt = fir_bandpass(4000, 10000, sr)

    # Filter audio
    bass = apply_filter(y, bass_filt)
    mid = apply_filter(y, mid_filt)
    treble = apply_filter(y, treble_filt)

    # Apply gain (convert dB to linear scale)
    bass *= 10**(bass_gain / 20)
    mid *= 10**(mid_gain / 20)
    treble *= 10**(treble_gain / 20)

    # Combine filtered signals
    equalized = bass + mid + treble
    equalized = equalized / np.max(np.abs(equalized))  # Normalize

    # Display waveform
    st.subheader("Waveform of Equalized Audio")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, equalized, color='purple')
    ax.set_title("Equalized Audio Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Save and playback
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as eq_audio:
        sf.write(eq_audio.name, equalized, sr)
        st.audio(eq_audio.name, format='audio/wav')

    os.unlink(filepath)
else:
    st.info("Upload a music or voice audio file to begin equalization.")
