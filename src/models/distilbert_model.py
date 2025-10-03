from transformers import DistilBertTokenizer, DistilBertModel as DistilBertBaseModel
import torch
import torch.nn as nn

class DistilBertModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2, num_features=6):
        super(DistilBertModel, self).__init__()
        self.distilbert = DistilBertBaseModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size + num_features, num_labels)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]  # CLS token
        combined = torch.cat((hidden_state, features), dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

    def train_model(self, texts, features, labels, epochs=3, batch_size=16, learning_rate=5e-5):
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from tqdm import tqdm

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

        dataset = EmailDataset(texts, features, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.train()

        for epoch in range(epochs):
            for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features_batch = batch['features'].to(device)
                labels_batch = batch['labels'].to(device)

                outputs = self(input_ids, attention_mask, features_batch)
                loss = loss_fn(outputs, labels_batch)
                loss.backward()
                optimizer.step()

    def predict(self, texts, features):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        predictions = []
        with torch.no_grad():
            for text, feat in zip(texts, features):
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=512,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                features_tensor = torch.tensor(feat, dtype=torch.float).unsqueeze(0).to(device)

                outputs = self(input_ids, attention_mask, features_tensor)
                _, predicted = torch.max(outputs, 1)
                predictions.append(predicted.item())

        return predictions

    def save_model(self, path):
        torch.save(self.state_dict(), f'{path}/distilbert_model.pth')
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.load_state_dict(torch.load(f'{path}/distilbert_model.pth'))
        self.tokenizer = DistilBertTokenizer.from_pretrained(path)
