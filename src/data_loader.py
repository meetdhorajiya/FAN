import os
import pandas as pd
import urllib.request

def download_etth1(data_dir="data"):
    """
    Downloads the ETTh1 dataset if it does not exist locally.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "ETTh1.csv")
    
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    
    if not os.path.exists(file_path):
        print(f"Downloading ETTh1 dataset from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
    else:
        print("Dataset already exists locally.")
        
    return file_path

def load_data(file_path):
    """
    Loads the dataset into a pandas DataFrame.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Data loaded successfully.")
    return df

def inspect_data(df):
    """
    Prints basic dataset information.
    """
    print("\n--- Dataset Inspection ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    print(f"Basic Statistics:\n{df.describe()}")
