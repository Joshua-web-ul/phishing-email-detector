import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib

# Load training data
train_data = pd.read_csv('data/preprocessed_train_data.csv')
emails = train_data['cleaned_text'].tolist()
labels = train_data['label'].tolist()

# Extract additional features for training
feature_columns = ['reply_to_mismatch', 'link_count', 'urgency_score', 'all_caps_subject', 'has_attachment', 'sender_domain_encoded']
features = train_data[feature_columns].values

# Prepare combined features
vectorizer = TfidfVectorizer()
scaler = StandardScaler()
X_text = vectorizer.fit_transform(emails)
X_features = scaler.fit_transform(features)
X = hstack([X_text, X_features])

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, labels)

# Best model
best_model = grid_search.best_estimator_
print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_:.4f}')

# Save best model, vectorizer, and scaler
joblib.dump(best_model, 'models/baseline_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(scaler, 'models/feature_scaler.pkl')

print('Tuned baseline model trained and saved.')
