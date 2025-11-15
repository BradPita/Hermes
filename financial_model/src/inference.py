"""
Inference Script for Financial Model

Author: eddy
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Optional

from .config import Config
from .model import create_model
from .dataset import create_sample_data, FinancialDataset

class Predictor:
    """Model predictor for inference"""
    
    def __init__(self, checkpoint_path: str, config: Optional[Config] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if config is None:
            self.config = checkpoint['config']
        else:
            self.config = config
        
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data
        
        Args:
            data: Input data of shape (batch_size, seq_length, input_dim)
                  or (seq_length, input_dim) for single prediction
        
        Returns:
            Predictions of shape (batch_size, pred_length) or (pred_length,)
        """
        single_sample = False
        if data.ndim == 2:
            data = data[np.newaxis, :]
            single_sample = True
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(data_tensor)
        
        predictions = predictions.cpu().numpy()
        
        if single_sample:
            predictions = predictions[0]
        
        return predictions
    
    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        features: Optional[list] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Make predictions from pandas DataFrame
        
        Args:
            df: DataFrame with financial data
            features: List of feature columns to use
            normalize: Whether to normalize the data
        
        Returns:
            Predictions array
        """
        if features is None:
            features = self.config.data.features
        
        dataset = FinancialDataset(
            df,
            seq_length=self.config.model.seq_length,
            pred_length=self.config.model.pred_length,
            features=features,
            target=self.config.data.target,
            normalize=normalize
        )
        
        predictions = []
        
        for i in range(len(dataset)):
            seq, _ = dataset[i]
            pred = self.predict(seq.numpy())
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if normalize and dataset.scaler_target is not None:
            predictions = dataset.inverse_transform_target(predictions)
        
        return predictions
    
    def predict_next(
        self,
        recent_data: pd.DataFrame,
        features: Optional[list] = None,
        normalize: bool = True
    ) -> float:
        """
        Predict next value based on recent data
        
        Args:
            recent_data: Recent data (should have at least seq_length rows)
            features: List of feature columns
            normalize: Whether to normalize
        
        Returns:
            Next predicted value
        """
        if features is None:
            features = self.config.data.features
        
        if len(recent_data) < self.config.model.seq_length:
            raise ValueError(
                f"Need at least {self.config.model.seq_length} rows, got {len(recent_data)}"
            )
        
        recent_data = recent_data.tail(self.config.model.seq_length)
        
        feature_data = recent_data[features].values
        
        if normalize:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            feature_data = scaler.fit_transform(feature_data)
        
        prediction = self.predict(feature_data)
        
        return prediction[0] if self.config.model.pred_length == 1 else prediction

def demo_inference():
    """Demo inference with trained model"""
    
    checkpoint_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    print("Loading model...")
    predictor = Predictor(checkpoint_path)
    
    print("\nCreating test data...")
    test_data = create_sample_data(num_samples=200)
    print(f"Test data shape: {test_data.shape}")
    print(test_data.head())
    
    print("\nMaking predictions...")
    predictions = predictor.predict_from_dataframe(test_data)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    
    print("\nPredicting next value...")
    next_value = predictor.predict_next(test_data)
    print(f"Next predicted value: {next_value:.2f}")
    print(f"Last actual value: {test_data['close'].iloc[-1]:.2f}")
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    demo_inference()

