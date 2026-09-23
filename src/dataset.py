import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class ETTDataset(Dataset):
    """Generic ETT Dataset — works for ETTh1, ETTh2, ETTm1, ETTm2."""
    def __init__(self, data_path, flag='train', seq_len=96, pred_len=24):
        assert flag in ['train', 'val', 'test']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag
        
        # Load data
        df = pd.read_csv(data_path)
        if df.empty:
            raise ValueError(f"Dataset is empty: {data_path}")
        
        # Preprocessing (similar to Phase 1)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(by='date').reset_index(drop=True)
            
        if df.isnull().sum().sum() > 0:
            df = df.ffill()

        feature_df = df.drop(columns=['date'], errors='ignore')
        if feature_df.empty:
            raise ValueError("Dataset must contain at least one numeric feature column.")
        try:
            feature_df = feature_df.apply(pd.to_numeric, errors='raise')
        except (TypeError, ValueError) as exc:
            raise ValueError("All feature columns must be numeric.") from exc
        if feature_df.isnull().any().any() or not np.isfinite(feature_df.values).all():
            raise ValueError("Dataset contains missing or non-finite values after preprocessing.")

        self.feature_names = feature_df.columns.tolist()
        features = feature_df.values
        
        # 7/2/1 split
        total_len = len(features)
        train_end = int(total_len * 0.7)
        val_end = int(total_len * 0.9)
        if train_end < seq_len + pred_len:
            raise ValueError(
                "Dataset is too short for the requested sequence and prediction lengths."
            )
        
        if flag == 'train':
            self.data = features[:train_end]
        elif flag == 'val':
            self.data = features[train_end - seq_len: val_end]
        elif flag == 'test':
            self.data = features[val_end - seq_len:]
            
        # Scaling - Fit strictly on train
        self.scaler = StandardScaler()
        train_data = features[:train_end]
        self.scaler.fit(train_data)
        
        self.data = self.scaler.transform(self.data)
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)
        
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

# Backward-compatible alias
ETTh1Dataset = ETTDataset

def get_dataloader(data_path, flag, seq_len, pred_len, batch_size=32):
    dataset = ETTDataset(data_path, flag, seq_len, pred_len)
    shuffle = True if flag == 'train' else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
