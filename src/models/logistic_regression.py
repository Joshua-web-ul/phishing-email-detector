from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import joblib
import os

class LogisticRegressionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        # Load pre-trained model if exists
        if os.path.exists('models/baseline_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
            self.load_model('models/baseline_model.pkl', 'models/tfidf_vectorizer.pkl')

    def train(self, emails, labels):
        X = self.vectorizer.fit_transform(emails)
        self.model.fit(X, labels)

    def predict(self, emails):
        print(f"DEBUG: predict called with emails type: {type(emails)}")
        # Ensure emails is iterable
        if isinstance(emails, str):
            emails = [emails]
        print(f"DEBUG: emails after wrapping if needed: {emails}")
        X = self.vectorizer.transform(emails)
        return self.model.predict(X)

    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path, vectorizer_path):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)