import pandas as pd
from src.models.bert_model import BertModel

# Load training data
train_data = pd.read_csv('data/preprocessed_train_data.csv')
emails = train_data['cleaned_text'].tolist()
labels = train_data['label'].tolist()

# Train model
model = BertModel()
model.train(emails, labels, epochs=1)  # Reduce epochs for demo

# Save model
model.save_model('models/fine_tuned_bert_model')

print('BERT model trained and saved.')
