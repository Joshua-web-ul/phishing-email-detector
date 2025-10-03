import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.utils.feature_extraction import add_features_to_dataframe

# Define function to clean email text
def clean_email(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\n', ' ', text)  # Remove newlines
    text = re.sub(r'\r', ' ', text)  # Remove carriage returns
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)  # Remove special characters
    return text.lower()  # Convert to lowercase

# Load Enron dataset
enron_path = 'data/enron/'
enron_files = [f for f in os.listdir(enron_path) if f.endswith('.csv')]
enron_data = pd.concat([pd.read_csv(os.path.join(enron_path, f)) for f in enron_files], ignore_index=True)

# Load phishing dataset
phishing_path = 'data/phishing/'
phishing_files = [f for f in os.listdir(phishing_path) if f.endswith('.csv')]
phishing_data = pd.concat([pd.read_csv(os.path.join(phishing_path, f)) for f in phishing_files], ignore_index=True)

# Keep original text for feature extraction
enron_data['original_text'] = enron_data['message'].fillna('')
phishing_data['original_text'] = phishing_data['Email Text'].fillna('') if 'Email Text' in phishing_data.columns else phishing_data.iloc[:, 0].fillna('')

# Clean the email text in both datasets
enron_data['cleaned_text'] = enron_data['message'].apply(clean_email).fillna('')
# Adjust phishing dataset column name for email text
if 'Email Text' in phishing_data.columns:
    phishing_data['cleaned_text'] = phishing_data['Email Text'].apply(clean_email).fillna('')
else:
    phishing_data['cleaned_text'] = phishing_data.iloc[:, 0].apply(clean_email).fillna('')  # Use first column as email text

# Combine datasets and create labels
enron_data['label'] = 0  # Legitimate emails
phishing_data['label'] = 1  # Phishing emails
combined_data = pd.concat([enron_data[['original_text', 'cleaned_text', 'label']], phishing_data[['original_text', 'cleaned_text', 'label']]], ignore_index=True)

# Extract additional features
combined_data = add_features_to_dataframe(combined_data, text_column='original_text')
print("Columns after feature extraction:", combined_data.columns.tolist())

# Drop rows with empty cleaned_text
combined_data = combined_data[combined_data['cleaned_text'] != '']

# Shuffle the combined dataset
combined_data = shuffle(combined_data, random_state=42)

# Split into training and testing sets
train_data, test_data = train_test_split(combined_data, test_size=0.2, random_state=42, stratify=combined_data['label'])

# Save the preprocessed data
train_data.to_csv('data/preprocessed_train_data.csv', index=False)
test_data.to_csv('data/preprocessed_test_data.csv', index=False)

print('Data preprocessing completed and saved to CSV files.')
