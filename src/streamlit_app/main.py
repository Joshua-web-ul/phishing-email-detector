import streamlit as st
import pandas as pd
import sys
import os
from functools import lru_cache
import hashlib
import random
import logging
from fpdf import FPDF
import requests

# Add project root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_loader import load_data
from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel
from src.models.ensemble_model import EnsembleModel
from src.utils.phishiness_score import PhishinessScore
from src.explainability.explain_why import ExplainWhy
from src.streamlit_app.components.explain_why_component import explain_why_component
from src.streamlit_app.components.mitigation_advice import display_mitigation_advice
from src.streamlit_app.components.footer import display_footer
from src.utils.feature_extraction import extract_features

# Custom CSS
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #1E88E5 0%, #ffffff 100%);
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
}
.phish-banner {
    background-color: #D32F2F;
    color: white;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
.safe-banner {
    background-color: #388E3C;
    color: white;
    padding: 10px;
    border-radius: 5px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}
.confidence-bar {
    width: 100%;
    height: 20px;
    background-color: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background-color: #1E88E5;
}
.awareness-tip {
    background-color: #FFF3CD;
    border: 1px solid #FFEAA7;
    border-radius: 5px;
    padding: 10px;
    margin: 10px 0;
}
.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 12px;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# Awareness tips
awareness_tips = [
    "Phishers often create urgency (e.g., ‘Act now!’).",
    "Check the reply-to email; it may differ from the sender.",
    "Never click links from unknown senders.",
    "Verify sender domains manually.",
    "Look for spelling errors in emails."
]

# Cache for predictions
@lru_cache(maxsize=100)
def cached_lr_predict(email_hash, email_text, feature_values):
    return logistic_model.predict([email_text], [feature_values])[0]

@lru_cache(maxsize=100)
def cached_bert_predict(email_hash, email_text, feature_values):
    return bert_model.predict([email_text], [feature_values])[0]

# Load models
logistic_model = LogisticRegressionModel()
bert_model = BertModel()
ensemble_model = EnsembleModel()

# Load data
enron_data, phishing_data = load_data()

# Header
col_header = st.container()
with col_header:
    st.markdown("""
    <div class="header">
        <h1>🛡️ AI-Powered Phishing Email Detector</h1>
        <p>Detect and understand suspicious emails — stay cyber aware.</p>
    </div>
    """, unsafe_allow_html=True)

# Disclaimer about data usage
st.info("Disclaimer: The data you input into this app may be used to improve and retrain the AI models to provide better phishing detection results in the future. By using this app, you consent to this data usage.")

# Input Section
st.header("📝 Input Email")

input_method = st.radio("Choose Input Method", ("📝 Paste Email Text", "📂 Upload .eml File", "📂 Batch Upload Multiple Emails"), horizontal=True)

email_text = ""
email_texts = []

if input_method == "📝 Paste Email Text":
    email_text = st.text_area("Paste your email content here…", height=200)
elif input_method == "📂 Upload .eml File":
    uploaded_file = st.file_uploader("Upload an email file (.txt or .eml)", type=["txt", "eml"])
    if uploaded_file is not None:
        email_text = uploaded_file.read().decode("utf-8")
elif input_method == "📂 Batch Upload Multiple Emails":
    uploaded_files = st.file_uploader("Upload multiple email files (.txt or .eml)", type=["txt", "eml"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            email_texts.append(file.read().decode("utf-8"))

# Large Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze Email", use_container_width=True, type="primary")

if analyze_button:
    if input_method == "Batch Upload Multiple Emails" and email_texts:
        with st.spinner('Analyzing batch emails...'):
            results = []
            for idx, email_text in enumerate(email_texts):
                extracted_features = extract_features(email_text)
                sender_domain_encoded = -1
                if 'sender_domain' in extracted_features:
                    sender_domain_encoded = hash(extracted_features['sender_domain']) % 10000
                feature_values = [extracted_features['reply_to_mismatch'], extracted_features['link_count'], extracted_features['urgency_score'], extracted_features['all_caps_subject'], extracted_features['has_attachment'], sender_domain_encoded]

                lr_prediction = logistic_model.predict([email_text], [feature_values])[0]
                bert_prediction = bert_model.predict([email_text], [feature_values])[0]
                ensemble_prediction = ensemble_model.predict([email_text], [feature_values])[0]

                results.append({
                    'Email #': idx + 1,
                    'Logistic Regression': "Phishing" if lr_prediction else "Legitimate",
                    'BERT': "Phishing" if bert_prediction else "Legitimate",
                    'Ensemble': "Phishing" if ensemble_prediction else "Legitimate"
                })

            st.write("Batch Analysis Results")
            st.dataframe(results)
    elif email_text.strip():
        with st.spinner('Analyzing email...'):
            # Extract features
            extracted_features = extract_features(email_text)
            # sender_domain_encoded is not directly available, encode it here
            sender_domain_encoded = -1
            if 'sender_domain' in extracted_features:
                sender_domain_encoded = hash(extracted_features['sender_domain']) % 10000  # simple hash encoding
            feature_values = [extracted_features['reply_to_mismatch'], extracted_features['link_count'], extracted_features['urgency_score'], extracted_features['all_caps_subject'], extracted_features['has_attachment'], sender_domain_encoded]

            # Predict using Logistic Regression
            lr_prediction = logistic_model.predict([email_text], [feature_values])[0]  # predict returns list
            # Predict using BERT
            bert_prediction = bert_model.predict([email_text], [feature_values])[0]
            # Predict using Ensemble
            ensemble_prediction = ensemble_model.predict([email_text], [feature_values])[0]

            # Calculate Phishiness Score (using combined features)
            from scipy.sparse import hstack
            text_features = logistic_model.vectorizer.transform([email_text])
            # Convert feature_values to sparse matrix and scale
            import numpy as np
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            numeric_features = np.array(feature_values).reshape(1, -1)
            numeric_features_scaled = scaler.fit_transform(numeric_features)
            from scipy.sparse import csr_matrix
            numeric_sparse = csr_matrix(numeric_features_scaled)
            combined_features = hstack([text_features, numeric_sparse])
            phishiness_calculator = PhishinessScore(logistic_model.model)
            phishiness_score = phishiness_calculator.calculate_score(combined_features)[0]

            # Display predictions in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Logistic Regression", "Phishing" if lr_prediction else "Legitimate",
                          delta=None,
                          delta_color="inverse" if lr_prediction else "normal")
            with col2:
                st.metric("BERT", "Phishing" if bert_prediction else "Legitimate",
                          delta=None,
                          delta_color="inverse" if bert_prediction else "normal")
            with col3:
                st.metric("Ensemble", "Phishing" if ensemble_prediction else "Legitimate",
                          delta=None,
                          delta_color="inverse" if ensemble_prediction else "normal")
            with col4:
                st.metric("Phishiness Score", f"{phishiness_score:.2f}")

            # Prediction Banner
            if ensemble_prediction:
                st.markdown('<div class="phish-banner">🚨 This email looks suspicious!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-banner">✅ No phishing indicators found</div>', unsafe_allow_html=True)

            # Confidence Score Bar
            confidence_percent = int(phishiness_score * 100)
            st.markdown(f'''
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence_percent}%;"></div>
                </div>
            ''', unsafe_allow_html=True)

            # Highlight risky words in email text
            risky_words = ["verify now", "urgent", "act now", "click here", "password", "account", "login", "bank", "invoice"]
            highlighted_text = email_text
            text_color = "#4CAF50" if ensemble_prediction == 0 else "#000000"  # Green for safe, black for phishing
            for word in risky_words:
                highlighted_text = highlighted_text.replace(word, f'<span style="color: #D32F2F; font-weight: bold;">{word}</span>')
            st.markdown(f"<div style='white-space: pre-wrap; color: {text_color};'>{highlighted_text}</div>", unsafe_allow_html=True)

            # Metadata warnings
            st.subheader("⚠️ Metadata Warnings")
            if extracted_features['reply_to_mismatch']:
                st.write("⚠️ Sender domain mismatch")
            if extracted_features['link_count'] > 0:
                st.write(f"🔗 {extracted_features['link_count']} suspicious link(s) detected")
            if extracted_features['has_attachment']:
                st.write("📎 Attachment flagged")

            # Why it was flagged (Top 3 reasons)
            st.subheader("Why it was flagged")
            reasons = []
            if extracted_features['urgency_score'] > 0.5:
                reasons.append("Urgency detected in email content")
            if extracted_features['reply_to_mismatch']:
                reasons.append("Reply-to address mismatch")
            if extracted_features['has_attachment']:
                reasons.append("Suspicious attachment present")
            for reason in reasons[:3]:
                st.write(f"• {reason}")

            # Awareness Tips Section
            st.markdown(f"<div class='awareness-tip'>💡 Cyber Awareness Tip: {random.choice(awareness_tips)}</div>", unsafe_allow_html=True)

            # User feedback
            st.subheader("Feedback")
            feedback = st.radio("Is the prediction correct?", ("Yes", "No"), key="feedback_radio")
            if feedback == "No":
                correct_label = st.selectbox("What is the correct label?", ("Legitimate", "Phishing"), key="correct_label")
                if st.button("Submit Feedback"):
                    # Save feedback
                    feedback_data = {
                        'email_text': email_text,
                        'predicted_label': ensemble_prediction,
                        'correct_label': 0 if correct_label == "Legitimate" else 1,
                        'features': feature_values
                    }
                    # Append to CSV
                    import csv
                    with open('data/user_feedback.csv', 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
                        if f.tell() == 0:  # File is empty, write header
                            writer.writeheader()
                        writer.writerow(feedback_data)
                    st.success("Feedback submitted! Thank you for helping improve the model.")
            else:
                st.write("Thank you for confirming the prediction.")

            # Save user input for future model improvement (optional)
            if st.button("Contribute to Model Improvement"):
                user_data = {
                    'original_text': email_text,
                    'cleaned_text': email_text,  # For simplicity, use original as cleaned
                    'label': int(ensemble_prediction),  # User can correct if wrong, but for now use prediction
                    'reply_to_mismatch': feature_values[0],
                    'link_count': feature_values[1],
                    'urgency_score': feature_values[2],
                    'all_caps_subject': feature_values[3],
                    'has_attachment': feature_values[4],
                    'sender_domain_encoded': feature_values[5]
                }
                import pandas as pd
                user_data_df = pd.DataFrame([user_data])
                user_data_df.to_csv('data/user_contributions.csv', mode='a', header=not os.path.exists('data/user_contributions.csv'), index=False)
                st.success("Thank you for contributing! Your data has been saved to help improve the model.")

            # Extra features
            st.subheader("Extra Features")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Download Report"):
                    if not email_text.strip():
                        st.warning("Please analyze an email first to download the report.")
                    else:
                        reasons = []
                        if extracted_features['urgency_score'] > 0.5:
                            reasons.append("Urgency detected in email content")
                        if extracted_features['reply_to_mismatch']:
                            reasons.append("Reply-to address mismatch")
                        if extracted_features['has_attachment']:
                            reasons.append("Suspicious attachment present")
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt="Phishing Email Detector Report", ln=True, align="C")
                        pdf.ln(10)
                        pdf.multi_cell(0, 10, txt=f"Email Text:\n{email_text}")
                        pdf.ln(5)
                        pdf.cell(0, 10, txt=f"Logistic Regression Prediction: {'Phishing' if lr_prediction else 'Legitimate'}", ln=True)
                        pdf.cell(0, 10, txt=f"BERT Prediction: {'Phishing' if bert_prediction else 'Legitimate'}", ln=True)
                        pdf.cell(0, 10, txt=f"Ensemble Prediction: {'Phishing' if ensemble_prediction else 'Legitimate'}", ln=True)
                        pdf.cell(0, 10, txt=f"Phishiness Score: {phishiness_score:.2f}", ln=True)
                        pdf.ln(5)
                        pdf.cell(0, 10, txt="Reasons for Flagging:", ln=True)
                        for reason in reasons:
                            pdf.cell(0, 10, txt=f"- {reason}", ln=True)
                        pdf_bytes = pdf.output(dest='S').encode('latin1')
                        st.download_button(label="Download PDF Report", data=pdf_bytes, file_name="phishing_report.pdf", mime="application/pdf")
            with col2:
                if st.button("🔗 Share on LinkedIn"):
                    if not email_text.strip():
                        st.warning("Please analyze an email first to share.")
                    else:
                        awareness_tip = random.choice(awareness_tips)
                        linkedin_url = f"https://www.linkedin.com/sharing/share-offsite/?url=https://phishing-detector.example.com&title={awareness_tip.replace(' ', '%20')}"
                        st.markdown(f'<a href="{linkedin_url}" target="_blank"><button>🔗 Share Awareness on LinkedIn</button></a>', unsafe_allow_html=True)
                        # Log share event
                        logging.info(f"User shared awareness tip: {awareness_tip}")

            # Log prediction event
            logging.info(f"Prediction: email_hash={hashlib.sha256(email_text.encode('utf-8')).hexdigest()}, lr={lr_prediction}, bert={bert_prediction}, ensemble={ensemble_prediction}, score={phishiness_score:.2f}")
    else:
        st.warning("Please enter or upload an email text to analyze.")

# Display footer on all pages
display_footer()

# Dark Mode CSS
if "dark_mode" in st.session_state and st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
        }
        .phish-banner {
            background-color: #b71c1c !important;
        }
        .safe-banner {
            background-color: #2e7d32 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Admin logging setup
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Footer
st.markdown("""
<div class="footer">
    <p>This tool is for educational use only. Always verify emails with official sources.</p>
    <p><a href="https://www.cybersecurityawareness.com/" target="_blank">Learn more about phishing → Cybersecurity Awareness Resources</a></p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")  # horizontal line for separation

st.markdown(
    """
    <div style="text-align: center;">
    Developed by: Joshua Muuo © 2025 
    </div> 
    <div style="text-align: center;">
    Feedback or Questions? <a href="mailto:joshua.miniprojects@gmail.com">Send me an email</a>
    </div>
    """,
    unsafe_allow_html=True
)
