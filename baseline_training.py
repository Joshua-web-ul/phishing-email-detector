import os
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
from scipy.sparse import hstack, csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
import textstat
from urllib.parse import urlparse
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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

class EmailFeatureExtractor:
    """Enhanced feature extraction for email classification"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_text_features(self, text):
        """Extract various text-based features"""
        if not isinstance(text, str):
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'stopword_ratio': 0,
                'readability_score': 0,
                'has_http': 0,
                'has_html': 0,
                'suspicious_words': 0
            }
            
        # Basic text features
        words = word_tokenize(text.lower())
        word_count = len(words)
        char_count = len(text)
        avg_word_length = char_count / max(1, word_count)
        
        # Stopword analysis
        stopword_count = sum(1 for word in words if word in self.stop_words)
        stopword_ratio = stopword_count / max(1, word_count)
        
        # Readability score
        readability = textstat.flesch_reading_ease(text)
        
        # Suspicious patterns
        has_http = 1 if 'http://' in text or 'https://' in text else 0
        has_html = 1 if bool(re.search(r'<[a-z][\s\S]*>', text)) else 0
        
        # Count suspicious words/phrases
        suspicious_phrases = [
            'urgent', 'verify', 'account', 'password', 'login', 'bank', 'paypal',
            'irs', 'lottery', 'prize', 'win', 'selected', 'click', 'update',
            'suspended', 'verify', 'confirm', 'ssn', 'social security', 'credit card'
        ]
        suspicious_count = sum(text.lower().count(phrase) for phrase in suspicious_phrases)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'stopword_ratio': stopword_ratio,
            'readability_score': readability,
            'has_http': has_http,
            'has_html': has_html,
            'suspicious_words': suspicious_count
        }
    
    def extract_url_features(self, text):
        """Extract URL-based features"""
        if not isinstance(text, str):
            return {
                'url_count': 0,
                'avg_url_length': 0,
                'url_shortened': 0,
                'url_has_ip': 0,
                'url_has_port': 0,
                'url_has_at': 0
            }
            
        # Find all URLs in text
        url_pattern = r'https?://[^\s\"]+'
        urls = re.findall(url_pattern, text)
        url_count = len(urls)
        
        if not urls:
            return {
                'url_count': 0,
                'avg_url_length': 0,
                'url_shortened': 0,
                'url_has_ip': 0,
                'url_has_port': 0,
                'url_has_at': 0
            }
        
        url_features = []
        for url in urls:
            try:
                parsed = urlparse(url)
                url_features.append({
                    'length': len(url),
                    'is_shortened': 1 if any(d in parsed.netloc for d in ['bit.ly', 'tinyurl', 'goo.gl']) else 0,
                    'has_ip': 1 if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', parsed.netloc) else 0,
                    'has_port': 1 if ':' in parsed.netloc else 0,
                    'has_at': 1 if '@' in url else 0
                })
            except:
                continue
                
        if not url_features:
            return {
                'url_count': url_count,
                'avg_url_length': 0,
                'url_shortened': 0,
                'url_has_ip': 0,
                'url_has_port': 0,
                'url_has_at': 0
            }
            
        # Aggregate URL features
        return {
            'url_count': url_count,
            'avg_url_length': sum(f['length'] for f in url_features) / len(url_features),
            'url_shortened': 1 if any(f['is_shortened'] for f in url_features) else 0,
            'url_has_ip': 1 if any(f['has_ip'] for f in url_features) else 0,
            'url_has_port': 1 if any(f['has_port'] for f in url_features) else 0,
            'url_has_at': 1 if any(f['has_at'] for f in url_features) else 0
        }

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
    feature_extractor = EmailFeatureExtractor()
    
    # Basic features
    features = []
    
    # Text features
    text_features = data['cleaned_text'].apply(lambda x: feature_extractor.extract_text_features(x)).apply(pd.Series)
    url_features = data['cleaned_text'].apply(lambda x: feature_extractor.extract_url_features(x)).apply(pd.Series)
    
    # Combine all features
    features = pd.concat([
        text_features,
        url_features,
        data[['reply_to_mismatch', 'link_count', 'urgency_score', 'all_caps_subject', 'has_attachment']]
    ], axis=1)
    
    logger.info(f"Extracted {features.shape[1]} features")
    return features

def train_model(X, y):
    """Train the model with cross-validation and hyperparameter tuning"""
    logger.info("Starting model training...")
    
    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            stop_words='english',
            sublinear_tf=True
        )),
        ('feature_selection', SelectKBest(chi2, k=5000)),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    # Define parameter grid for tuning
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_features': [5000, 10000, 20000],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
    
    # Use StratifiedKFold for imbalanced datasets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    logger.info("Starting grid search...")
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

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
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Load and preprocess data
        data = load_and_preprocess_data('data/preprocessed_train_data.csv')
        
        # Extract features
        X = data['cleaned_text']
        y = data['label']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)
        
        # Save the model
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else []
        save_model(model, model.named_steps['tfidf'], None, feature_names)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
