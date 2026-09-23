import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


class NeuroSignatureModel:
    """
    One-class EEG neuro-signature model.

    The model learns the distribution of available EEG feature samples
    and identifies whether a new sample is consistent with that
    learned neuro-signature profile.
    """

    def __init__(
        self,
        n_estimators=200,
        contamination=0.05,
        random_state=42
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def train(self, X):
        """
        Train the one-class neuro-signature model.
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "Input features must be a 2-dimensional array."
            )

        self.model.fit(X)

        return self

    def predict(self, X):
        """
        Predict whether samples belong to the learned EEG profile.

        Returns:
            1  -> accepted / in-distribution
           -1  -> rejected / anomalous
        """

        X = np.asarray(X, dtype=np.float64)

        return self.model.predict(X)

    def decision_score(self, X):
        """
        Return anomaly decision scores.

        Higher values indicate samples that are more consistent
        with the learned training distribution.
        """

        X = np.asarray(X, dtype=np.float64)

        return self.model.decision_function(X)

    def save(self, path):
        """Save the trained model."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a trained model."""
        self.model = joblib.load(path)
