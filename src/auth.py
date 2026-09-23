import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class NeuroAuthSystem:
    """
    EEG-based neuro-signature enrollment and verification system.

    The system creates a reference signature from enrollment samples
    and verifies a new sample using cosine similarity.
    """

    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.templates = {}

    def enroll_user(self, user_id, eeg_samples):
        """
        Create a neuro-signature template from enrollment samples.

        Parameters
        ----------
        user_id : str
            Identifier for the enrolled user.

        eeg_samples : array-like
            Multiple preprocessed EEG feature vectors.
        """

        eeg_samples = np.asarray(eeg_samples, dtype=np.float64)

        if eeg_samples.ndim != 2:
            raise ValueError(
                "EEG samples must be a 2-dimensional array."
            )

        if len(eeg_samples) < 2:
            raise ValueError(
                "At least two enrollment samples are recommended."
            )

        # Average the enrollment feature vectors
        template = np.mean(eeg_samples, axis=0)

        self.templates[user_id] = template

        return template

    def authenticate(self, user_id, test_sample):
        """
        Verify a test EEG feature vector against the enrolled template.

        Returns
        -------
        authenticated : bool
            Whether the similarity meets the threshold.

        similarity : float
            Cosine similarity between the template and test sample.
        """

        if user_id not in self.templates:
            raise ValueError(
                f"User '{user_id}' is not enrolled."
            )

        test_sample = np.asarray(
            test_sample,
            dtype=np.float64
        ).reshape(1, -1)

        template = self.templates[user_id].reshape(1, -1)

        similarity = cosine_similarity(
            template,
            test_sample
        )[0][0]

        authenticated = similarity >= self.threshold

        return authenticated, float(similarity)

    def set_threshold(self, threshold):
        """
        Update the authentication threshold.
        """

        if not 0 <= threshold <= 1:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def get_threshold(self):
        """Return the current authentication threshold."""
        return self.threshold
