from shap import Explainer
import numpy as np
import pandas as pd

class ExplainWhy:
    def __init__(self, model, vectorizer, shap_explainer):
        self.model = model
        self.vectorizer = vectorizer
        self.shap_explainer = shap_explainer

    def explain_email(self, email_text):
        # Transform the email text into the feature vector
        email_vector = self.vectorizer.transform([email_text])
        
        # Get the SHAP values
        shap_values = self.shap_explainer.shap_values(email_vector)

        # Get the prediction
        prediction = self.model.predict(email_vector)

        # Prepare the explanation
        explanation = self._prepare_explanation(email_text, shap_values)

        return {
            "prediction": prediction[0],
            "explanation": explanation
        }

    def _prepare_explanation(self, email_text, shap_values):
        # Get feature names from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Create a DataFrame for SHAP values
        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        # Get the absolute SHAP values and sort them
        shap_abs = np.abs(shap_df).mean(axis=0)
        sorted_indices = np.argsort(shap_abs)[::-1]

        # Prepare the explanation text
        explanation = []
        for idx in sorted_indices[:10]:  # Top 10 features
            explanation.append((feature_names[idx], shap_abs[idx]))

        return explanation