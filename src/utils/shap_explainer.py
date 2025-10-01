import shap
import joblib
import numpy as np

class ShapExplainer:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def explain(self, text):
        # Transform the text
        text_vectorized = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()

        # Use SHAP LinearExplainer for LogisticRegression
        explainer = shap.LinearExplainer(self.model, text_vectorized)
        shap_values = explainer.shap_values(text_vectorized)

        # Get the SHAP values for the positive class (phishing)
        shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

        # Get top contributing words
        top_positive = np.argsort(shap_vals)[-5:][::-1]  # Top 5 positive
        top_negative = np.argsort(shap_vals)[:5]  # Top 5 negative

        explanation = {
            'top_positive_words': [(feature_names[i], shap_vals[i]) for i in top_positive],
            'top_negative_words': [(feature_names[i], shap_vals[i]) for i in top_negative]
        }

        return explanation
