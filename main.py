import os
import numpy as np

from src.preprocessing import load_eeg_features, scale_features
from src.auth import NeuroAuthSystem


DATA_PATH = "data/raw/emotions_positive_only.csv"


def main():
    print("=" * 60)
    print("Neuro-Signature-Based Authentication System")
    print("=" * 60)

    # 1. Check dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # 2. Load EEG-derived features
    X, labels = load_eeg_features(DATA_PATH)

    print(f"\nDataset samples: {X.shape[0]}")
    print(f"EEG feature count: {X.shape[1]}")
    print(f"Labels: {np.unique(labels)}")

    # 3. Scale features
    X_scaled, scaler = scale_features(X)

    print("Feature preprocessing completed.")

    # 4. Split enrollment and verification samples
    enrollment_size = int(len(X_scaled) * 0.80)

    enrollment_samples = X_scaled[:enrollment_size]
    verification_samples = X_scaled[enrollment_size:]

    print(
        f"Enrollment samples: {len(enrollment_samples)}"
    )
    print(
        f"Verification samples: {len(verification_samples)}"
    )

    # 5. Create authentication system
    auth_system = NeuroAuthSystem(
        contamination=0.05,
        n_estimators=200
    )

    # 6. Enroll reference neuro-signature
    user_id = "user_001"

    auth_system.enroll_user(
        user_id,
        enrollment_samples
    )

    print(
        f"\nNeuro-signature profile created "
        f"for '{user_id}'."
    )

    # 7. Verify samples
    print("\nVerification Results")
    print("-" * 45)

    accepted = 0
    rejected = 0

    for index, sample in enumerate(
        verification_samples[:10],
        start=1
    ):
        authenticated, score = auth_system.authenticate(
            sample
        )

        if authenticated:
            status = "IN-PROFILE"
            accepted += 1
        else:
            status = "ANOMALOUS"
            rejected += 1

        print(
            f"Sample {index:02d}: "
            f"{status} | "
            f"Score: {score:.4f}"
        )

    print("\nSummary")
    print("-" * 45)
    print(f"Accepted / In-profile: {accepted}")
    print(f"Rejected / Anomalous:  {rejected}")

    print(
        "\nNote: This is a one-class verification "
        "prototype. The dataset does not contain "
        "multiple subject IDs or impostor samples."
    )


if __name__ == "__main__":
    main()
