import os
import numpy as np
from src.preprocessing import load_emotion_data, preprocess_eeg, segment_data
from src.feature_extraction import extract_features
from src.model import EmotionAuthModel

def main():
    # 1. Load and preprocess data
    data_path = "data/raw/emotions_positive_only.csv"
    eeg_data, labels = load_emotion_data(data_path)
    processed_data = preprocess_eeg(eeg_data)

    # 2. Create segments and extract features
    segments = segment_data(processed_data)
    X = np.array([extract_features(seg) for seg in segments])
    y = labels[:len(segments)]  # ✅ Fixed: removed np.labels

    # 3. Train model
    model = EmotionAuthModel(model_type='rf')
    model.train(X, y)
    model.save("emotion_auth_model.joblib")

    # 4. Example prediction
    test_sample = X[0]
    prediction = model.model.predict([test_sample])[0]
    print(f"Sample prediction: {prediction}")

if __name__ == "__main__":
    main()
