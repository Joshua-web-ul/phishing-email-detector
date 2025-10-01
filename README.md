# Phishing Email Detector with Explainable AI

## Motivation
In today's digital age, phishing attacks have become increasingly sophisticated, posing significant threats to individuals and organizations. The Phishing Email Detector project aims to provide a robust solution for identifying phishing emails using machine learning techniques. By leveraging the Enron dataset for legitimate emails and publicly available phishing datasets, this project not only detects phishing attempts but also explains the reasoning behind its predictions, enhancing user trust and understanding.

## Project Overview
This project utilizes a combination of traditional machine learning and advanced deep learning techniques. The baseline model employs TF-IDF vectorization with Logistic Regression, while a fine-tuned BERT model is used for more nuanced analysis. The integration of SHAP (SHapley Additive exPlanations) allows users to understand which features influenced the model's predictions.

## Unique Features
- **Explain Why**: Users can see which words or phrases contributed to the classification of an email as phishing.
- **Phishiness Score**: A score that quantifies the likelihood of an email being a phishing attempt.
- **Safe Mitigation Advice**: Provides users with actionable steps to take if an email is flagged as phishing.

## Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/phishing-email-detector.git
   cd phishing-email-detector
   ```

2. **Install Requirements**
   It is recommended to create a virtual environment before installing the dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Datasets**
   Place the Enron dataset files in the `data/enron` directory and the public phishing dataset files in the `data/phishing` directory.

4. **Run the Streamlit App**
   Start the Streamlit application using the following command:
   ```bash
   streamlit run src/streamlit_app/main.py
   ```

## Usage
- **Home Page**: Upload an email to check if it is phishing or legitimate.
- **Explain Why**: After classification, click on the "Explain Why" button to see the highlighted words that influenced the decision.
- **Phishiness Score**: View the calculated score indicating the likelihood of the email being a phishing attempt.
- **Mitigation Advice**: Get recommendations on how to handle the flagged email.

## Screenshots
- **Home Page**: ![Home Page](screenshots/home_page.png)
- **Explain Why Page**: ![Explain Why Page](screenshots/explain_why_page.png)
- **Phishiness Score Page**: ![Phishiness Score Page](screenshots/phishiness_score_page.png)

## Conclusion
The Phishing Email Detector with Explainable AI is a comprehensive tool designed to enhance email security. By combining machine learning with explainability, it empowers users to make informed decisions about their email communications.
