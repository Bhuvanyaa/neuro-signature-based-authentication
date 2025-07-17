import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt

def extract_features(segment):
    """Extract multi-domain features from EEG segment"""
    features = []
    
    # Time-domain features
    for channel in segment.T:
        features.extend([
            np.mean(channel), np.std(channel),
            skew(channel), kurtosis(channel),
            np.median(channel), np.max(channel)-np.min(channel)
        ])
    
    # Frequency-domain features
    for channel in segment.T:
        freqs, psd = welch(channel, 256, nperseg=64)
        for band in [(1,4), (4,8), (8,13), (13,30), (30,40)]:
            idx = (freqs >= band[0]) & (freqs <= band[1])
            features.append(np.log(np.sum(psd[idx]) + 1e-12))
    
    # Wavelet features
    for channel in segment.T:
        coeffs = pywt.wavedec(channel, 'db4', level=4)
        for coeff in coeffs:
            features.extend([
                np.mean(coeff), np.std(coeff),
                np.sum(coeff**2)
            ])
    
    return np.array(features)