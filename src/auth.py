import numpy as np

from src.model import NeuroSignatureModel


class NeuroAuthSystem:
    """
    EEG neuro-signature verification system using
    one-class anomaly detection.
    """

    def __init__(
        self,
        contamination=0.05,
        n_estimators=200,
        random_state=42
    ):
        self.model = NeuroSignatureModel(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

        self.user_id = None
        self.enrolled = False

    def enroll_user(self, user_id, eeg_samples):
        """
        Create a neuro-signature profile from enrollment samples.
        """

        eeg_samples = np.asarray(
            eeg_samples,
            dtype=np.float64
        )

        if eeg_samples.ndim != 2:
            raise ValueError(
                "EEG samples must be a 2-dimensional array."
            )

        if len(eeg_samples) < 10:
            raise ValueError(
                "At least 10 EEG samples are required for enrollment."
            )

        self.model.train(eeg_samples)

        self.user_id = user_id
        self.enrolled = True

        return True

    def authenticate(self, test_sample):
        """
        Verify a new EEG feature sample against
        the learned neuro-signature profile.
        """

        if not self.enrolled:
            raise ValueError(
                "No user has been enrolled."
            )

        test_sample = np.asarray(
            test_sample,
            dtype=np.float64
        ).reshape(1, -1)

        prediction = self.model.predict(test_sample)[0]

        score = self.model.decision_score(
            test_sample
        )[0]

        authenticated = prediction == 1

        return authenticated, float(score)

    def save(self, path):
        """Save the trained neuro-signature model."""
        self.model.save(path)

    def load(self, path):
        """Load a previously trained neuro-signature model."""
        self.model.load(path)
        self.enrolled = True