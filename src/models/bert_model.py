from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BertModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def train(self, train_texts, train_labels, epochs=3, batch_size=16, learning_rate=5e-5):
        from torch.utils.data import DataLoader, Dataset
        from torch.optim import AdamW
        from tqdm import tqdm

        class EmailDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=512):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
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
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        dataset = EmailDataset(train_texts, train_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(dataloader, leave=True)
            for batch in loop:
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                loop.set_description(f'Epoch {epoch + 1}/{epochs}')
                loop.set_postfix(loss=loss.item())

    def predict(self, texts):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            # Batch process texts for efficiency
            inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions = preds.tolist()
        return predictions

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)