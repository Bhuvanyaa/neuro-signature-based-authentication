import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NeuroSignatureModel:
    """
    Machine-learning model for EEG-derived neuro-signature features.

    This model provides a classification component for the project.
    The authentication layer is handled separately by NeuroAuthSystem.
    """

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.classes_ = None

    def train(self, X, y):
        """
        Train the Random Forest model and report test accuracy.

        Note:
        A meaningful multi-class identity model requires samples from
        multiple subjects/classes. If the dataset contains only one
        class, classification accuracy is not a useful authentication
        metric.
        """

        unique_classes = np.unique(y)

        if len(unique_classes) < 2:
            print(
                "Only one class is present in the dataset. "
                "Skipping classifier training."
            )
            self.classes_ = unique_classes
            return None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Random Forest Test Accuracy: {accuracy:.4f}")

        return accuracy

    def predict(self, X):
        """Predict the class for one or more feature vectors."""
        if self.model is None:
            raise ValueError("Model has not been trained.")

        return self.model.predict(X)

    def save(self, path):
        """Save the trained model."""
        joblib.dump(self.model, path)

    def load(self, path):
        """Load a previously trained model."""
        self.model = joblib.load(path)
        self.classes_ = getattr(self.model, "classes_", None)
