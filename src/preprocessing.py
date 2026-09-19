import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col='OT', train_ratio=0.7, val_ratio=0.2):
    """
    Preprocesses the data: parses timestamp, handles missing values,
    splits into train/val/test, and normalizes based on training data.
    """
    print("\n--- Data Preprocessing ---")
    
    # 1. Timestamp parsing
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        # Sort by date just in case
        df = df.sort_values(by='date').reset_index(drop=True)
    
    # 2. Missing value check/fill (forward fill as basic strategy)
    if df.isnull().sum().sum() > 0:
        print("Missing values found. Filling with forward fill.")
        df = df.ffill()
        
    # Drop date for scaling/modeling purposes, keep features
    features = df.drop(columns=['date'])
    
    # 3. Train/Validation/Test split (7/2/1)
    total_len = len(features)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    train_data = features.iloc[:train_end]
    val_data = features.iloc[train_end:val_end]
    test_data = features.iloc[val_end:]
    
    print(f"Split sizes -> Train: {train_data.shape}, Validation: {val_data.shape}, Test: {test_data.shape}")
    
    # 4. Normalization/Scaling (Fit ONLY on training data to avoid data leakage)
    scaler = StandardScaler()
    scaler.fit(train_data)
    
    train_scaled = pd.DataFrame(scaler.transform(train_data), columns=train_data.columns)
    val_scaled = pd.DataFrame(scaler.transform(val_data), columns=val_data.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
    
    print("Scaling complete. Sample processed training data:")
    print(train_scaled.head())
    
    return train_scaled, val_scaled, test_scaled, scaler
