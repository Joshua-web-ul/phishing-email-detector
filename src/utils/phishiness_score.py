from sklearn.preprocessing import MinMaxScaler
import numpy as np

class PhishinessScore:
    def __init__(self, model):
        self.model = model
        self.scaler = MinMaxScaler()

    def calculate_score(self, features):
        # Get the prediction probabilities from the model
        probabilities = self.model.predict_proba(features)[:, 1]
        # Scale the probabilities to a score between 0 and 100
        scaled_scores = self.scaler.fit_transform(probabilities.reshape(-1, 1))
        return scaled_scores.flatten() * 100

    def interpret_score(self, score):
        if score < 30:
            return "Low risk of phishing."
        elif 30 <= score < 70:
            return "Moderate risk of phishing. Exercise caution."
        else:
            return "High risk of phishing! Immediate action required."

def main():
    # Example usage
    # Load your model and features here
    # model = load_model('path_to_model')
    # features = extract_features(email_text)

    # phishiness_score_calculator = PhishinessScore(model)
    # score = phishiness_score_calculator.calculate_score(features)
    # interpretation = phishiness_score_calculator.interpret_score(score)

    pass  # This is intentionally left blank for the main function.