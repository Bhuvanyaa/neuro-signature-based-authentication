import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
import pywt

def load_emotion_data(file_path):
    """Load emotion EEG data from CSV"""
    df = pd.read_csv(file_path)
    
    # Assuming format: [timestep, channel1, channel2, ..., emotion_label]
    eeg_data = df.iloc[:, :-1].values  # All columns except last
    labels = df.iloc[:, -1].values     # Last column is emotion label
    
    return eeg_data.astype(np.float64), labels

def bandpass_filter(data, sfreq=256, l_freq=1, h_freq=40):
    """Butterworth bandpass filter"""
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)

def preprocess_eeg(eeg_data):
    """Full preprocessing pipeline"""
    # Filtering
    filtered = bandpass_filter(eeg_data)
    
    # Remove artifacts
    filtered = filtered[~np.isnan(filtered).any(axis=1)]
    
    # Standardize
    return StandardScaler().fit_transform(filtered)

def segment_data(data, window_size=256, overlap=0.5):
    """Create sliding windows"""
    step = int(window_size * (1 - overlap))
    return np.array([data[i:i+window_size] 
                   for i in range(0, len(data)-window_size+1, step)])