import pandas as pd
import optuna
import torch
from torch.utils.data import DataLoader, Dataset
from src.models.bert_model import BertModel

class EmailDataset(Dataset):
    def __init__(self, texts, features, labels, tokenizer, max_length=512):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        feature = torch.tensor(self.features[idx], dtype=torch.float)
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': feature,
            'labels': torch.tensor(label, dtype=torch.long)
        }

def objective(trial):
    train_data = pd.read_csv('data/preprocessed_train_data.csv')
    emails = train_data['cleaned_text'].tolist()
    labels = train_data['label'].tolist()
    feature_columns = ['reply_to_mismatch', 'link_count', 'urgency_score', 'all_caps_subject', 'has_attachment', 'sender_domain_encoded']
    features = train_data[feature_columns].values

    model_name = 'bert-base-uncased'
    num_labels = 2
    model = BertModel(model_name=model_name, num_labels=num_labels, num_features=features.shape[1])
    tokenizer = model.tokenizer

    dataset = EmailDataset(emails, features, labels, tokenizer)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    epochs = trial.suggest_int('epochs', 2, 5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features_batch = batch['features'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, features_batch)
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()

    # Evaluate on training data for simplicity (can add validation split)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features_batch = batch['features'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, features_batch)
            _, predicted = torch.max(outputs, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    accuracy = correct / total
    return accuracy

def train_best_model():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print(f'Best hyperparameters: {best_params}')

    # Train final model with best params
    train_data = pd.read_csv('data/preprocessed_train_data.csv')
    emails = train_data['cleaned_text'].tolist()
    labels = train_data['label'].tolist()
    feature_columns = ['reply_to_mismatch', 'link_count', 'urgency_score', 'all_caps_subject', 'has_attachment', 'sender_domain_encoded']
    features = train_data[feature_columns].values

    model_name = 'bert-base-uncased'
    num_labels = 2
    model = BertModel(model_name=model_name, num_labels=num_labels, num_features=features.shape[1])
    tokenizer = model.tokenizer

    dataset = EmailDataset(emails, features, labels, tokenizer)
    batch_size = best_params['batch_size']
    epochs = best_params['epochs']
    learning_rate = best_params['learning_rate']

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features_batch = batch['features'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, features_batch)
            loss = loss_fn(outputs, labels_batch)
            loss.backward()
            optimizer.step()

    model.save_model('models/fine_tuned_bert_model')
    print('BERT model trained and saved.')

if __name__ == '__main__':
    train_best_model()
