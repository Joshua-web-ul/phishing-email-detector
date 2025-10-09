 from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
import os
import numpy as np
from scipy.sparse import hstack

class LogisticRegressionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.scaler = StandardScaler()
        self.model = LogisticRegression()
        # Load pre-trained model if exists
        model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        model_path = os.path.join(model_dir, 'baseline_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and os.path.exists(scaler_path):
            self.load_model(model_path, vectorizer_path, scaler_path)

    def train(self, emails, features, labels):
        # Remove any NaN or None emails before training
        clean_data = [(email, feat, label) for email, feat, label in zip(emails, features, labels) if isinstance(email, str) and email.strip()]
        if not clean_data:
            raise ValueError("No valid emails to train on after cleaning.")
        clean_emails, clean_features, clean_labels = zip(*clean_data)
        X_text = self.vectorizer.fit_transform(clean_emails)
        X_features = np.array(clean_features)
        X_features = self.scaler.fit_transform(X_features)
        X = hstack([X_text, X_features])
        self.model.fit(X, clean_labels)

    def predict(self, emails, features):
        print(f"DEBUG: predict called with emails type: {type(emails)}")
        # Ensure emails is iterable
        if isinstance(emails, str):
            emails = [emails]
            features = [features]
        print(f"DEBUG: emails after wrapping if needed: {emails}")
        X_text = self.vectorizer.transform(emails)
        X_features = np.array(features)
        X_features = self.scaler.transform(X_features)
        X = hstack([X_text, X_features])
        preds = self.model.predict(X)
        # Convert string labels to numeric if needed
        if preds.dtype.type is np.str_:
            preds = [1 if p.lower() == 'phishing' else 0 for p in preds]
        return preds

    def save_model(self, model_path, vectorizer_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path, vectorizer_path, scaler_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.scaler = joblib.load(scaler_path)
