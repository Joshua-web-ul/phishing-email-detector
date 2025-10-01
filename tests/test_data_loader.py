import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.data_loader import load_data

def test_load_data():
    enron_data, phishing_data = load_data()
    assert isinstance(enron_data, pd.DataFrame), "Enron data should be a DataFrame"
    assert isinstance(phishing_data, pd.DataFrame), "Phishing data should be a DataFrame"
    assert not enron_data.empty, "Enron dataset should not be empty"
    assert not phishing_data.empty, "Phishing dataset should not be empty"
    assert 'email' in enron_data.columns, "Enron DataFrame should contain 'email' column"
    assert 'label' in enron_data.columns, "Enron DataFrame should contain 'label' column"
    assert 'email' in phishing_data.columns, "Phishing DataFrame should contain 'email' column"
    assert 'label' in phishing_data.columns, "Phishing DataFrame should contain 'label' column"
