import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models.logistic_regression import LogisticRegressionModel
from src.models.bert_model import BertModel

def evaluate():
    # Load test data
    test_data = pd.read_csv('data/preprocessed_test_data.csv')
    emails = test_data['cleaned_text'].tolist()
    labels = test_data['label'].tolist()

    # Extract additional features for testing
    feature_columns = ['reply_to_mismatch', 'link_count', 'urgency_score', 'all_caps_subject', 'has_attachment', 'sender_domain_encoded']
    features = test_data[feature_columns].values.tolist()

    # Initialize models
    logistic_model = LogisticRegressionModel()
    bert_model = BertModel()

    # Predict with Logistic Regression
    logistic_predictions = logistic_model.predict(emails, features)

    # Predict with BERT
    bert_predictions = bert_model.predict(emails, features)

    # Evaluate Logistic Regression
    logistic_accuracy = accuracy_score(labels, logistic_predictions)
    logistic_precision = precision_score(labels, logistic_predictions)
    logistic_recall = recall_score(labels, logistic_predictions)
    logistic_f1 = f1_score(labels, logistic_predictions)

    # Evaluate BERT
    bert_accuracy = accuracy_score(labels, bert_predictions)
    bert_precision = precision_score(labels, bert_predictions)
    bert_recall = recall_score(labels, bert_predictions)
    bert_f1 = f1_score(labels, bert_predictions)

    # Print results
    print("Logistic Regression Model Evaluation:")
    print(f"Accuracy: {logistic_accuracy:.4f}")
    print(f"Precision: {logistic_precision:.4f}")
    print(f"Recall: {logistic_recall:.4f}")
    print(f"F1 Score: {logistic_f1:.4f}")

    print("\nBERT Model Evaluation:")
    print(f"Accuracy: {bert_accuracy:.4f}")
    print(f"Precision: {bert_precision:.4f}")
    print(f"Recall: {bert_recall:.4f}")
    print(f"F1 Score: {bert_f1:.4f}")


if __name__ == "__main__":
    evaluate()
