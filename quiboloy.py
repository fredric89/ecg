import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt
import io

st.title("1D ECG Signal Filtering App")
st.markdown("Remove baseline wander and powerline noise from ECG data.")

st.sidebar.header("Upload ECG Data")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Bandpass filter settings
lowcut = 0.5
highcut = 40.0
fs = st.sidebar.number_input("Sampling Rate (Hz)", min_value=50, max_value=1000, value=250)
order = 4

@st.cache_data
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

if file is not None:
    # Load data
    df = pd.read_csv(file)
    column_names = df.columns.tolist()
    ecg_col = st.selectbox("Select ECG Column", column_names)
    ecg_data = df[ecg_col].values

    # Apply filtering
    filtered_ecg = butter_bandpass_filter(ecg_data, lowcut, highcut, fs, order)

    # Time axis
    t = np.arange(len(ecg_data)) / fs

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, ecg_data, label='Original ECG', color='gray')
    ax[0].set_title("Original ECG Signal")
    ax[0].set_ylabel("Amplitude")
    ax[0].grid(True)

    ax[1].plot(t, filtered_ecg, label='Filtered ECG', color='green')
    ax[1].set_title("Filtered ECG Signal (0.5–40 Hz Bandpass)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude")
    ax[1].grid(True)

    st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Commentary")
    st.markdown("**QRS Visibility:** The QRS complexes should appear sharper and more prominent in the filtered ECG. The baseline drift and 50/60 Hz powerline interference are reduced, making features like R-peaks more discernible.")
else:
    st.info("Upload an ECG CSV file to begin.")
