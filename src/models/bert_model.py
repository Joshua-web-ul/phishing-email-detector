from transformers import BertTokenizer, BertModel as BertBaseModel, BertConfig
import torch
import torch.nn as nn

class BertModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_labels=2, num_features=6):
        super(BertModel, self).__init__()
        self.bert = BertBaseModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size + num_features, num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined = torch.cat((pooled_output, features), dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits

    def train_model(self, train_texts, train_features, train_labels, epochs=3, batch_size=16, learning_rate=5e-5):
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
                features = self.features[idx]
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
                    'features': torch.tensor(features, dtype=torch.float),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        dataset = EmailDataset(train_texts, train_features, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.train()
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                features = batch['features']
                labels = batch['labels']

                logits = self(input_ids, attention_mask, features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                loop.set_description(f'Epoch {epoch + 1}/{epochs}')
                loop.set_postfix(loss=loss.item())

    def predict(self, texts, features):
        self.eval()
        predictions = []
        with torch.no_grad():
            # Batch process texts for efficiency
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            features = torch.tensor(features, dtype=torch.float)
            logits = self(inputs['input_ids'], inputs['attention_mask'], features)
            preds = torch.argmax(logits, dim=1)
            predictions = preds.tolist()
        return predictions

    def save_model(self, path):
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.save_pretrained(path)

    def load_model(self, path):
        self.load_state_dict(torch.load(f"{path}/pytorch_model.bin"))
        self.tokenizer = BertTokenizer.from_pretrained(path)
