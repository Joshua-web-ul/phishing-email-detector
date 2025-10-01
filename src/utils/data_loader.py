import pandas as pd
import os

def load_enron_data(data_dir):
    """
    Load the Enron dataset from the specified directory.
    
    Args:
        data_dir (str): The directory containing the Enron dataset files.
        
    Returns:
        pd.DataFrame: A DataFrame containing the Enron emails.
    """
    enron_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not enron_files:
        return pd.DataFrame()
    enron_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in enron_files], ignore_index=True)
    enron_data = enron_data.rename(columns={'message': 'email'})
    enron_data['label'] = 'legitimate'
    return enron_data[['email', 'label']]

def load_phishing_data(data_dir):
    """
    Load the phishing dataset from the specified directory.
    
    Args:
        data_dir (str): The directory containing the phishing dataset files.
        
    Returns:
        pd.DataFrame: A DataFrame containing the phishing emails.
    """
    phishing_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not phishing_files:
        return pd.DataFrame()
    phishing_data = pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in phishing_files], ignore_index=True)
    phishing_data = phishing_data.rename(columns={phishing_data.columns[0]: 'email'})
    phishing_data['label'] = 'phishing'
    return phishing_data[['email', 'label']]

def load_data():
    """
    Load both Enron and phishing datasets and return them separately.
    
    Returns:
        tuple: (enron_data, phishing_data) as DataFrames.
    """
    enron_data = load_enron_data('data/enron')
    phishing_data = load_phishing_data('data/phishing')
    return enron_data, phishing_data
