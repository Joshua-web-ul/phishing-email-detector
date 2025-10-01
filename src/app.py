from streamlit import st
import pandas as pd
from src.utils.data_loader import load_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel
from src.utils.phishiness_score import calculate_phishiness_score
from src.explainability.explain_why import explain_prediction

# Load models
logistic_model = LogisticRegressionModel()
bert_model = BertModel()

# Load data
enron_data, phishing_data = load_data()

# Streamlit app title
st.title("Phishing Email Detector with Explainable AI")

# User input for email text
email_text = st.text_area("Enter the email text to analyze:")

if st.button("Analyze"):
    # Predict using Logistic Regression
    log_pred = logistic_model.predict(email_text)
    bert_pred = bert_model.predict(email_text)

    # Calculate Phishiness Score
    phishiness_score = calculate_phishiness_score(email_text)

    # Display predictions
    st.subheader("Predictions")
    st.write(f"Logistic Regression Prediction: {'Phishing' if log_pred else 'Legitimate'}")
    st.write(f"BERT Prediction: {'Phishing' if bert_pred else 'Legitimate'}")
    st.write(f"Phishiness Score: {phishiness_score:.2f}")

    # Explain why
    explanation = explain_prediction(email_text, log_pred)
    st.subheader("Explanation")
    st.write(explanation)

    # Mitigation advice
    st.subheader("Mitigation Advice")
    if log_pred or bert_pred:
        st.write("This email is likely a phishing attempt. Do not click on any links or provide personal information.")
    else:
        st.write("This email appears to be legitimate. However, always verify the sender's email address.")

# Footer
st.markdown("### Note: This application is for educational purposes only. Always exercise caution with emails.")