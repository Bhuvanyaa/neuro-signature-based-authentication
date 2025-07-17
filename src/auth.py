import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NeuroAuthSystem:
    def __init__(self, model_path=None, threshold=0.9):
        self.model_path = model_path
        self.threshold = threshold
        self.templates = {}  # user_id: template_features
        
    def enroll_user(self, user_id, eeg_samples):
        """Enroll a new user by creating a template from their EEG samples"""
        # Average features across samples to create a template
        template = np.mean(eeg_samples, axis=0)
        self.templates[user_id] = template
        return template
    
    def authenticate(self, user_id, test_sample):
        """Authenticate a user by comparing with stored template"""
        if user_id not in self.templates:
            raise ValueError(f"User {user_id} not enrolled")
            
        template = self.templates[user_id]
        
        # Calculate similarity (can use different metrics)
        similarity = cosine_similarity([template], [test_sample])[0][0]
        
        return similarity >= self.threshold, similarity
    
    def adaptive_threshold(self, new_samples, user_id=None):
        """Adjust threshold based on new samples"""
        if user_id:
            # User-specific threshold adjustment
            template = self.templates[user_id]
            similarities = [cosine_similarity([template], [sample])[0][0] for sample in new_samples]
        else:
            # Global threshold adjustment
            similarities = []
            for uid, template in self.templates.items():
                for sample in new_samples:
                    sim = cosine_similarity([template], [sample])[0][0]
                    similarities.append(sim)
        
        # Set threshold to mean - 2*std of genuine scores (example)
        self.threshold = np.mean(similarities) - 2*np.std(similarities)
        return self.threshold