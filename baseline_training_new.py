import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve, auc)
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.feature_extraction import extract_features as extract_email_features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('reports', exist_ok=True)

def load_and_preprocess_data(filepath):
    """Load and preprocess the training data"""
    logger.info(f"Loading data from {filepath}")
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} samples")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def extract_features(data):
    """Extract features from the dataset"""
    logger.info("Extracting features...")
    features_df = data['cleaned_text'].apply(extract_email_features).apply(pd.Series)
    features_df['sender_domain_encoded'] = features_df['sender_domain'].astype('category').cat.codes
    features_df = features_df.drop('sender_domain', axis=1)
    logger.info(f"Extracted {features_df.shape[1]} features")
    return features_df

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and generate reports"""
    logger.info("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Log metrics
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])
    logger.info("\nClassification Report:\n" + report)

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # Generate and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('reports/roc_curve.png')
    plt.close()

def save_model(model, vectorizer, scaler, feature_names):
    """Save the trained model and related artifacts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f'models/model_{timestamp}'
    os.makedirs(model_dir, exist_ok=True)

    # Save model components
    joblib.dump(model, f'{model_dir}/model.pkl')
    joblib.dump(vectorizer, f'{model_dir}/vectorizer.pkl')
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')

    # Save feature names
    with open(f'{model_dir}/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))

    logger.info(f"Model and artifacts saved to {model_dir}")
    return model_dir

def main():
    try:
        # Load and preprocess data
        data = load_and_preprocess_data('data/preprocessed_train_data.csv')

        # Extract features
        features_df = extract_features(data)

        # Prepare X_text
        X_text = data['cleaned_text']
        y = data['label']

        # Split data
        X_text_train, X_text_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
            X_text, features_df, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit vectorizer
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english', sublinear_tf=True)
        X_text_train_vec = vectorizer.fit_transform(X_text_train)
        X_text_test_vec = vectorizer.transform(X_text_test)

        # Fit scaler
        scaler = StandardScaler()
        X_feat_train_scaled = scaler.fit_transform(X_feat_train)
        X_feat_test_scaled = scaler.transform(X_feat_test)

        # Combine
        X_train = hstack([X_text_train_vec, X_feat_train_scaled])
        X_test = hstack([X_text_test_vec, X_feat_test_scaled])

        # Train model
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        evaluate_model(model, X_test, y_test)

        # Save to expected paths
        joblib.dump(model, 'models/baseline_model.pkl')
        joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(scaler, 'models/feature_scaler.pkl')

        # Also save to timestamped
        save_model(model, vectorizer, scaler, features_df.columns.tolist())

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
