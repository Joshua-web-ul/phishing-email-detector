import pytest
from src.models.logistic_regression import LogisticRegressionModel
from src.utils.data_loader import load_data

@pytest.fixture
def setup_data():
    # Load the Enron dataset and phishing dataset
    enron_data, phishing_data = load_data()
    emails = list(enron_data['email']) + list(phishing_data['email'])
    labels = list(enron_data['label']) + list(phishing_data['label'])
    return emails, labels

@pytest.fixture
def setup_model(setup_data):
    emails, labels = setup_data
    # Initialize and train the Logistic Regression model
    model = LogisticRegressionModel()
    model.train(emails, labels)
    return model

def test_model_training(setup_model):
    model = setup_model
    assert model is not None
    assert hasattr(model, 'model')
    assert hasattr(model, 'vectorizer')

def test_model_prediction(setup_model):
    model = setup_model
    sample_email = ["Congratulations! You've won a lottery. Click here to claim your prize."]
    prediction = model.predict(sample_email)
    assert prediction[0] in [0, 1]  # Assuming 0 for legitimate and 1 for phishing

def test_model_save_load(setup_model):
    model = setup_model
    model.save_model('models/test_model.pkl', 'models/test_vectorizer.pkl')
    loaded_model = LogisticRegressionModel()
    loaded_model.load_model('models/test_model.pkl', 'models/test_vectorizer.pkl')
    assert loaded_model is not None
    assert hasattr(loaded_model, 'model')
    assert hasattr(loaded_model, 'vectorizer')
