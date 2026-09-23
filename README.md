# Neuro-Signature-Based Authentication

An EEG-based biometric authentication prototype that learns a neuro-signature profile from EEG-derived features and performs similarity-independent anomaly-based verification using Isolation Forest.

## Overview

Traditional authentication systems commonly rely on passwords, PINs, or physical biometrics. This project explores EEG-based biometric authentication by using brainwave-derived features as an authentication signal.

The system processes pre-extracted EEG features, learns the distribution of enrollment samples, and evaluates new EEG feature samples using one-class anomaly detection.

## System Architecture

```text
EEG Feature Dataset
        |
        v
Feature Loading & Cleaning
        |
        v
Feature Standardization
        |
        v
Enrollment Samples
        |
        v
Isolation Forest Model
        |
        v
Learned Neuro-Signature Profile
        |
        v
New EEG Feature Sample
        |
        v
Anomaly Detection
        |
   +----+----+
   |         |
   v         v
IN-PROFILE   ANOMALOUS
```

## Key Features

* EEG-derived biometric authentication prototype
* Pre-extracted EEG feature processing
* Missing and infinite value handling
* Feature standardization using StandardScaler
* One-class anomaly detection
* Isolation Forest-based verification
* Configurable contamination parameter
* Modular Python implementation
* Reproducible project structure

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* SciPy
* PyWavelets
* Joblib

## Project Structure

```text
neuro-signature-based-authentication/
|
├── data/
│   └── raw/
│       └── emotions_positive_only.csv
│
├── src/
│   ├── auth.py
│   ├── feature_extraction.py
│   ├── model.py
│   └── preprocessing.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

### 1. Feature Loading

The system loads the pre-extracted EEG feature dataset from CSV format.

The dataset contains multiple categories of EEG-derived numerical features, including statistical, FFT, covariance, eigenvalue, entropy, and correlation features.

### 2. Feature Preprocessing

The preprocessing pipeline:

* Removes the `label` column from the feature matrix
* Keeps numerical EEG features
* Replaces infinite values
* Handles missing values using feature medians
* Standardizes the feature vectors using `StandardScaler`

### 3. Neuro-Signature Enrollment

During enrollment, 80% of the available EEG feature samples are used to learn the reference neuro-signature profile.

The Isolation Forest model learns the distribution of the enrollment samples.

### 4. Verification

A new EEG feature sample is passed to the trained model.

The model produces:

* `IN-PROFILE` — the sample is considered consistent with the learned distribution
* `ANOMALOUS` — the sample is considered an outlier

The model also produces a decision score.

Higher decision scores indicate greater consistency with the learned training distribution.

## Machine Learning Approach

This project uses **Isolation Forest** for one-class anomaly detection.

Isolation Forest is useful when the available dataset does not contain separate positive and negative identity classes.

The model learns patterns from the available enrollment samples and identifies samples that appear unusual relative to that learned distribution.

```text
Enrollment EEG Features
          |
          v
   Isolation Forest
          |
          v
 Learned EEG Profile
          |
          v
New EEG Feature Sample
          |
          v
  Anomaly Detection
          |
     +----+----+
     |         |
     v         v
IN-PROFILE   ANOMALOUS
```

## Model Configuration

The current prototype uses:

```text
n_estimators = 200
contamination = 0.05
random_state = 42
```

The `contamination` parameter represents the expected proportion of anomalous observations used by the model.

These settings are prototype parameters and have not been established as biometric security standards.

## Dataset

The current project uses a pre-extracted EEG feature dataset.

Dataset characteristics:

```text
Samples: 708
EEG-derived features: 2,548
Label column: label
Available label: POSITIVE
```

The dataset contains EEG-derived numerical features rather than raw EEG signal recordings.

## Important Limitation

The current dataset contains only a single label class and does not provide multiple subject/user identifiers or separate impostor samples.

Therefore, this project should **not** be interpreted as a validated multi-user biometric identification system.

The current implementation demonstrates the technical workflow for:

```text
EEG Features
     |
     v
Enrollment Profile
     |
     v
One-Class Anomaly Detection
     |
     v
Verification Decision
```

The `user_001` identifier used by the demo is a conceptual enrollment identifier and does not represent a verified real-world subject identity.

A future version should use subject-labelled EEG data containing multiple users and genuine/impostor samples to properly evaluate biometric authentication performance.

## Security Considerations

EEG signals are biometric information and should be handled carefully.

A production implementation should consider:

* Secure storage of biometric templates
* Encryption at rest and in transit
* Template protection
* Replay-attack resistance
* Liveness detection
* Threshold and model calibration
* False Acceptance Rate (FAR)
* False Rejection Rate (FRR)
* Equal Error Rate (EER)
* Secure user enrollment
* Privacy and informed consent requirements

## Future Enhancements

* Streamlit authentication dashboard
* Subject-labelled multi-user EEG dataset
* PCA-based dimensionality reduction
* Advanced feature selection
* Model and parameter calibration
* FAR/FRR evaluation
* ROC curve analysis
* EEG template protection
* Liveness detection
* Real-time EEG authentication
* Secure API integration

## Installation

Clone the repository:

```bash
git clone https://github.com/Bhuvanyaa/neuro-signature-based-authentication.git
cd neuro-signature-based-authentication
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Execute:

```bash
python main.py
```

The application loads the EEG feature dataset, preprocesses the features, trains the one-class neuro-signature model, and verifies new samples against the learned distribution.

## Example Output

```text
============================================================
Neuro-Signature-Based Authentication System
============================================================

Dataset samples: 708
EEG feature count: 2548
Labels: ['POSITIVE']

Feature preprocessing completed.
Enrollment samples: 566
Verification samples: 142

Neuro-signature profile created for 'user_001'.

Verification Results
---------------------------------------------
Sample 01: IN-PROFILE | Score: 0.0957
Sample 02: IN-PROFILE | Score: 0.1226
Sample 03: IN-PROFILE | Score: 0.1877
...
```

The exact decision scores may vary depending on the model configuration and execution environment.

## Disclaimer

This project is an academic cybersecurity and machine-learning prototype intended for educational and research purposes.

It should not be considered a production-ready biometric authentication system.

## Author

**Bhuvanyaa S.**

Cybersecurity Student
B.Sc. Computer Science with Cybersecurity
