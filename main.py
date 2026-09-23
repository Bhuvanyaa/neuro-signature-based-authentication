import os
import numpy as np

from src.preprocessing import load_eeg_features, scale_features
from src.auth import NeuroAuthSystem


DATA_PATH = "data/raw/emotions_positive_only.csv"


def main():
    print("=" * 60)
    print("Neuro-Signature-Based Authentication System")
    print("=" * 60)

    # 1. Load pre-extracted EEG features
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    X, labels = load_eeg_features(DATA_PATH)

    print(f"\nDataset samples: {X.shape[0]}")
    print(f"EEG feature count: {X.shape[1]}")
    print(f"Labels: {np.unique(labels)}")

    # 2. Scale EEG features
    X_scaled, scaler = scale_features(X)

    print("Feature preprocessing completed.")

    # 3. Create authentication system
    auth_system = NeuroAuthSystem(threshold=0.90)

    # 4. Use part of the available samples for enrollment
    enrollment_size = int(len(X_scaled) * 0.8)

    enrollment_samples = X_scaled[:enrollment_size]
    test_samples = X_scaled[enrollment_size:]

    # 5. Enroll the reference neuro-signature
    user_id = "user_001"

    auth_system.enroll_user(
        user_id,
        enrollment_samples
    )

    print(
        f"\nUser '{user_id}' enrolled successfully."
    )

    # 6. Verify test samples
    print("\nAuthentication Results")
    print("-" * 40)

    for index, sample in enumerate(test_samples[:5], start=1):

        authenticated, similarity = auth_system.authenticate(
            user_id,
            sample
        )

        status = (
            "AUTHENTICATED"
            if authenticated
            else "REJECTED"
        )

        print(
            f"Test Sample {index}: "
            f"{status} | "
            f"Similarity: {similarity:.4f}"
        )


if __name__ == "__main__":
    main()
