import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

class EmotionAuthModel:
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.classes_ = None
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True)
        
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{self.model_type.upper()} Test Accuracy: {acc:.4f}")
        
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
        self.classes_ = self.model.classes_