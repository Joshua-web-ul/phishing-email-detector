from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel
import numpy as np

class EnsembleModel:
    def __init__(self):
        self.lr_model = LogisticRegressionModel()
        self.bert_model = BertModel()

    def predict(self, emails, features):
        # Get predictions from both models
        lr_preds = self.lr_model.predict(emails, features)
        bert_preds = self.bert_model.predict(emails, features)

        # Voting: if both agree, use that; else use LR (or majority, but since 2, use LR)
        ensemble_preds = []
        for lr, bert in zip(lr_preds, bert_preds):
            if lr == bert:
                ensemble_preds.append(lr)
            else:
                ensemble_preds.append(lr)  # Default to LR, or could use random or other logic

        return ensemble_preds

    def save_model(self, path):
        # Save both models
        self.lr_model.save_model(f'{path}/lr_model.pkl', f'{path}/lr_vectorizer.pkl', f'{path}/lr_scaler.pkl')
        self.bert_model.save_model(f'{path}/bert_model')

    def load_model(self, path):
        self.lr_model.load_model(f'{path}/lr_model.pkl', f'{path}/lr_vectorizer.pkl', f'{path}/lr_scaler.pkl')
        self.bert_model.load_model(f'{path}/bert_model')
