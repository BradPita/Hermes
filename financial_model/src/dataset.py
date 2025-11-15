"""
Financial Dataset

Author: eddy
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class FinancialDataset(Dataset):
    """Financial time series dataset"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        seq_length: int,
        pred_length: int,
        features: list,
        target: str,
        normalize: bool = True
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.features = features
        self.target = target
        
        self.feature_data = data[features].values
        self.target_data = data[target].values
        
        self.scaler_features = StandardScaler() if normalize else None
        self.scaler_target = StandardScaler() if normalize else None
        
        if normalize:
            self.feature_data = self.scaler_features.fit_transform(self.feature_data)
            self.target_data = self.scaler_target.fit_transform(self.target_data.reshape(-1, 1)).flatten()
        
        self.sequences = []
        self.targets = []
        
        for i in range(len(self.feature_data) - seq_length - pred_length + 1):
            seq = self.feature_data[i:i + seq_length]
            target = self.target_data[i + seq_length:i + seq_length + pred_length]
            
            self.sequences.append(seq)
            self.targets.append(target)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )
    
    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform target data"""
        if self.scaler_target is not None:
            return self.scaler_target.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

def create_sample_data(num_samples: int = 1000) -> pd.DataFrame:
    """Create sample financial data for testing"""
    
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=num_samples, freq='D')
    
    price = 100
    prices = []
    
    for _ in range(num_samples):
        change = np.random.randn() * 2
        price = price * (1 + change / 100)
        prices.append(price)
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(num_samples) * 0.5,
        'high': prices + np.abs(np.random.randn(num_samples)) * 1.0,
        'low': prices - np.abs(np.random.randn(num_samples)) * 1.0,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, num_samples)
    })
    
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data

def create_dataloaders(
    data: pd.DataFrame,
    config,
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    n = len(data)
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    train_dataset = FinancialDataset(
        train_data,
        config.model.seq_length,
        config.model.pred_length,
        config.data.features,
        config.data.target,
        config.data.normalize
    )
    
    val_dataset = FinancialDataset(
        val_data,
        config.model.seq_length,
        config.model.pred_length,
        config.data.features,
        config.data.target,
        config.data.normalize
    )
    
    test_dataset = FinancialDataset(
        test_data,
        config.model.seq_length,
        config.model.pred_length,
        config.data.features,
        config.data.target,
        config.data.normalize
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler_target

