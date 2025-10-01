from sklearn.linear_model import LogisticRegression
import shap
import pandas as pd
import numpy as np
import joblib

class SHAPExplainer:
    def __init__(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.explainer = shap.Explainer(self.model)

    def explain(self, email_text):
        # Transform the email text into the feature vector
        email_vector = self.vectorizer.transform([email_text])
        
        # Get the SHAP values
        shap_values = self.explainer(email_vector)
        
        return shap_values

    def get_feature_importance(self, shap_values):
        # Get the feature importance from SHAP values
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        return feature_importance

    def get_feature_names(self):
        # Get feature names from the vectorizer
        return self.vectorizer.get_feature_names_out()

    def explain_email(self, email_text):
        shap_values = self.explain(email_text)
        feature_importance = self.get_feature_importance(shap_values)
        feature_names = self.get_feature_names()

        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })

        # Sort the DataFrame by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        return importance_df.head(10), shap_values

# Example usage:
# explainer = SHAPExplainer('models/baseline_model.pkl', 'models/tfidf_vectorizer.pkl')
# top_features, shap_values = explainer.explain_email("Your email text here")