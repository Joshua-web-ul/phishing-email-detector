import streamlit as st
import pandas as pd
import sys
import os

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_loader import load_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel
from src.utils.phishiness_score import PhishinessScore
from src.explainability.explain_why import ExplainWhy
from src.streamlit_app.components.explain_why_component import explain_why
from src.streamlit_app.components.mitigation_advice import display_mitigation_advice

# Load models
logistic_model = LogisticRegressionModel()
bert_model = BertModel()

# Load data
enron_data, phishing_data = load_data()

# Streamlit app title
st.title("Phishing Email Detector with Explainable AI")

# Container for input and button
with st.container():
    email_text = st.text_area("Enter the email text to analyze:", height=200)
    analyze_button = st.button("Analyze")

if analyze_button:
    if email_text:
        with st.spinner('Analyzing email...'):
            # Predict using Logistic Regression
            lr_prediction = logistic_model.predict(email_text)
            # Predict using BERT
            bert_prediction = bert_model.predict(email_text)

            # Calculate Phishiness Score
            features = logistic_model.extract_features([email_text])
            phishiness_calculator = PhishinessScore(logistic_model.model)
            phishiness_score = phishiness_calculator.calculate_score(features)[0]

            # Display predictions in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Logistic Regression", "Phishing" if lr_prediction else "Legitimate",
                          delta=None,
                          delta_color="inverse" if lr_prediction else "normal")
            with col2:
                st.metric("BERT", "Phishing" if bert_prediction else "Legitimate",
                          delta=None,
                          delta_color="inverse" if bert_prediction else "normal")
            with col3:
                st.metric("Phishiness Score", f"{phishiness_score:.2f}")

            # Explain why
            explain_why = ExplainWhy(logistic_model.model, logistic_model.vectorizer, logistic_model.shap_explainer)
            explanation = explain_why.explain_email(email_text)
            explain_why(explanation['prediction'], explanation['explanation'], logistic_model.vectorizer.get_feature_names_out(), email_text)

            # Mitigation advice
            phishing_result = lr_prediction or bert_prediction
            display_mitigation_advice(phishing_result)
    else:
        st.warning("Please enter an email text to analyze.")
