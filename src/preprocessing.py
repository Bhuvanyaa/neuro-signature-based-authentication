import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_eeg_features(file_path):
    """
    Load pre-extracted EEG features from a CSV file.

    The dataset contains EEG-derived features such as statistical,
    FFT, covariance, eigenvalue, entropy, and correlation features.
    The final 'label' column is excluded from the feature matrix.
    """
    df = pd.read_csv(file_path)

    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X = df.drop(columns=["label"])
    y = df["label"]

    # Keep only numeric EEG features
    X = X.select_dtypes(include=[np.number])

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values using column medians
    X = X.fillna(X.median())

    return X.to_numpy(dtype=np.float64), y.to_numpy()


def scale_features(X):
    """
    Standardize EEG features before similarity comparison or modeling.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
