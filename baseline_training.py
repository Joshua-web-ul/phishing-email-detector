import pandas as pd
from src.models.logistic_regression import LogisticRegressionModel

# Load training data
train_data = pd.read_csv('data/preprocessed_train_data.csv')
emails = train_data['cleaned_text'].tolist()
labels = train_data['label'].tolist()

# Train model
model = LogisticRegressionModel()
model.train(emails, labels)

# Save model and vectorizer
model.save_model('models/baseline_model.pkl', 'models/tfidf_vectorizer.pkl')

print('Baseline model trained and saved.')
