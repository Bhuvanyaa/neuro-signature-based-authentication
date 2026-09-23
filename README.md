# Neuro-Signature-Based Authentication

An EEG-based biometric authentication prototype that creates a reference neuro-signature from EEG-derived features and verifies new samples using cosine similarity.

## Overview

Traditional authentication systems commonly rely on passwords, PINs, or physical biometrics. This project explores EEG-based biometric authentication by using brainwave-derived features as a potential authentication signal.

The system processes pre-extracted EEG features, creates a reference neuro-signature during enrollment, and compares new EEG feature samples against the stored signature.

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
Neuro-Signature Enrollment
        |
        v
Reference EEG Template
        |
        v
New EEG Feature Sample
        |
        v
Cosine Similarity
        |
        v
Similarity Threshold
        |
   +----+----+
   |         |
   v         v
AUTHENTICATED  REJECTED


 

**Key Features**

EEG-derived biometric authentication
Feature cleaning and preprocessing
Feature standardization using StandardScaler
Neuro-signature template generation
Cosine similarity-based verification
Configurable authentication threshold
Modular Python implementation
Reproducible project structure
Technologies
Python
NumPy
Pandas
Scikit-learn
SciPy
PyWavelets
Joblib
Project Structure
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
How It Works
1. Feature Loading

The system loads the pre-extracted EEG feature dataset from CSV format.

The dataset contains multiple categories of EEG-derived features, including statistical, FFT, covariance, eigenvalue, entropy, and correlation features.

2. Feature Preprocessing

The preprocessing pipeline:

Removes the label column from the feature matrix
Keeps numerical EEG features
Replaces infinite values
Handles missing values using feature medians
Standardizes the feature vectors
3. Neuro-Signature Enrollment

During enrollment, multiple EEG feature samples are used to create a reference template.

The reference template is calculated from the enrollment samples and stored for subsequent verification.

4. Authentication

A new EEG feature sample is compared with the enrolled template using cosine similarity.

If the similarity score meets the configured threshold, the sample is accepted.

Otherwise, the authentication attempt is rejected.

Authentication Formula

Cosine similarity measures the similarity between two feature vectors based on their orientation in feature space.

similarity = cosine(reference_signature, test_signature)

The current prototype uses:

Threshold = 0.90

This value is a configurable prototype setting and has not been established as a security-standard threshold.

Dataset

The current project uses a pre-extracted EEG feature dataset.

The dataset contains EEG-derived numerical features rather than raw EEG signal recordings.

The current dataset contains a single POSITIVE label class. Therefore, this version of the project demonstrates neuro-signature verification rather than multi-user identity classification.

Important Limitation

The current dataset does not provide multiple subject/user identifiers.

Therefore, this project should not be interpreted as a validated multi-user biometric identification system.

The current implementation demonstrates the technical workflow for:

EEG Features
     |
     v
Reference Neuro-Signature
     |
     v
Similarity-Based Verification

A future version can use subject-labelled EEG data to evaluate multi-user identification and authentication performance.

Security Considerations

EEG signals are biometric information and should be handled carefully.

A production implementation should consider:

Secure storage of biometric templates
Encryption at rest and in transit
Template protection
Replay-attack resistance
Liveness detection
Threshold calibration
False Acceptance Rate (FAR)
False Rejection Rate (FRR)
Equal Error Rate (EER)
Secure user enrollment
Privacy and consent requirements
Future Enhancements
Streamlit authentication dashboard
Subject-labelled multi-user EEG dataset
Advanced feature selection
PCA-based dimensionality reduction
Authentication threshold calibration
FAR/FRR evaluation
ROC curve analysis
EEG template encryption
Liveness detection
Real-time EEG authentication
Secure API integration
Installation

Clone the repository:

git clone https://github.com/Bhuvanyaa/neuro-signature-based-authentication.git
cd neuro-signature-based-authentication

Install dependencies:

pip install -r requirements.txt
Run

Execute:

python main.py

The application loads the EEG feature dataset, preprocesses the features, creates a reference neuro-signature, and performs verification against test samples.

Example Output
============================================================
Neuro-Signature-Based Authentication System
============================================================

Dataset samples: 708
EEG feature count: 2548
Labels: ['POSITIVE']

Feature preprocessing completed.

User 'user_001' enrolled successfully.

Authentication Results
----------------------------------------
Test Sample 1: AUTHENTICATED | Similarity: 0.XXXX
Test Sample 2: AUTHENTICATED | Similarity: 0.XXXX

The similarity values depend on the dataset and preprocessing performed during execution.

Disclaimer

This project is an academic cybersecurity and machine-learning prototype. It is intended for educational and research purposes and should not be considered a production-ready biometric authentication system.

Author

Bhuvanyaa S.

Cybersecurity Student
B.Sc. computer science with Cybersecurity
