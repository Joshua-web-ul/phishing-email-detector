from transformers import BertTokenizer, BertModel
import torch

class BertEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def generate_embeddings(self, texts):
        # Tokenize the input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # The embeddings are the last hidden states
        embeddings = outputs.last_hidden_state
        
        # Return the mean of the embeddings for each input text
        return embeddings.mean(dim=1)  # Shape: (batch_size, hidden_size)

# Example usage:
# bert_embeddings = BertEmbeddings()
# embeddings = bert_embeddings.generate_embeddings(["This is a test email.", "Please click this link."])