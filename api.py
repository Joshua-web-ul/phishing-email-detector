from flask import Flask, request, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel
from src.models.ensemble_model import EnsembleModel
from src.utils.feature_extraction import extract_features

app = Flask(__name__)

# Load models
logistic_model = LogisticRegressionModel()
bert_model = BertModel()
ensemble_model = EnsembleModel()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email_text', '')
    if not email_text:
        return jsonify({'error': 'No email text provided'}), 400

    # Extract features
    extracted_features = extract_features(email_text)
    sender_domain_encoded = -1
    if 'sender_domain' in extracted_features:
        sender_domain_encoded = hash(extracted_features['sender_domain']) % 10000
    feature_values = [extracted_features['reply_to_mismatch'], extracted_features['link_count'], extracted_features['urgency_score'], extracted_features['all_caps_subject'], extracted_features['has_attachment'], sender_domain_encoded]

    # Predictions
    lr_prediction = logistic_model.predict([email_text], [feature_values])[0]
    bert_prediction = bert_model.predict([email_text], [feature_values])[0]
    ensemble_prediction = ensemble_model.predict([email_text], [feature_values])[0]

    return jsonify({
        'logistic_regression': 'Phishing' if lr_prediction else 'Legitimate',
        'bert': 'Phishing' if bert_prediction else 'Legitimate',
        'ensemble': 'Phishing' if ensemble_prediction else 'Legitimate'
    })

if __name__ == '__main__':
    app.run(debug=True)
