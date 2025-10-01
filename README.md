AI-Powered Phishing Email Detector with Explainability  

Overview  
Phishing remains one of the biggest cybersecurity threats, tricking users into revealing sensitive information through deceptive emails.  
This project leverages Artificial Intelligence (AI) + Natural Language Processing (NLP) to detect phishing emails and explain why they are risky.  

Unlike most phishing detectors that give a simple yes/no output, this project:  
- Highlights risky words and patterns** (e.g., urgent, verify account).  
- Uses metadata features (sender domain, link counts, caps lock subject lines).  
- Provides confidence scores and top reasons behind classification.  

The goal: Detect phishing + raise awareness by teaching users how to spot them.  

 Features  
- AI Model trained on real phishing + legitimate emails.  
- Hybrid Features: Text (TF-IDF, BERT embeddings) + Metadata (domains, links).  
- Explainable AI (XAI): Uses LIME/SHAP to show why an email is phishing.  
- Web App (Streamlit):Paste email text or upload `.eml` file → Get prediction.  
- Educational Mode: Shows cybersecurity awareness tips alongside results.  

Dataset  
This project combines ham (legit) emails from the Enron dataset with phishing datasets from Kaggle.  

- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)  
- [Kaggle Phishing Email Dataset](https://www.kaggle.com/)  

Tech Stack  
- Python 3.9+
- Libraries: 
  - `scikit-learn` → Logistic Regression & Random Forest baseline  
  - `transformers` (HuggingFace) → DistilBERT for NLP  
  - `lime` & `shap` → Explainability  
  - `pandas`, `numpy`, `matplotlib` → Data handling  
  - `streamlit` → Web UI  

cd phishing-email-detector

