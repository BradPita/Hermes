"""
Improved Loss Functions for Financial Prediction

Author: eddy
"""

import torch
import torch.nn as nn


class DirectionalLoss(nn.Module):
    """
    Loss that penalizes incorrect direction predictions
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        target_diff = targets[:, 1:] - targets[:, :-1]
        
        direction_correct = (pred_diff * target_diff) > 0
        direction_loss = 1.0 - direction_correct.float().mean()
        
        return direction_loss


class TrendAwareLoss(nn.Module):
    """
    Combined loss that encourages both accuracy and trend prediction
    
    Components:
    1. MSE: Basic prediction accuracy
    2. Directional: Correct trend direction
    3. Magnitude: Correct change magnitude
    """
    def __init__(self, mse_weight=0.5, direction_weight=0.3, magnitude_weight=0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        
        self.mse = nn.MSELoss()
        self.directional = DirectionalLoss()
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        
        if predictions.shape[1] > 1:
            direction_loss = self.directional(predictions, targets)
            
            pred_changes = torch.abs(predictions[:, 1:] - predictions[:, :-1])
            target_changes = torch.abs(targets[:, 1:] - targets[:, :-1])
            magnitude_loss = self.mse(pred_changes, target_changes)
            
            total_loss = (
                self.mse_weight * mse_loss +
                self.direction_weight * direction_loss +
                self.magnitude_weight * magnitude_loss
            )
        else:
            total_loss = mse_loss
        
        return total_loss


class HuberTrendLoss(nn.Module):
    """
    Huber loss variant that is less sensitive to outliers
    and encourages trend prediction
    """
    def __init__(self, delta=1.0, trend_weight=0.3):
        super().__init__()
        self.delta = delta
        self.trend_weight = trend_weight
        self.huber = nn.HuberLoss(delta=delta)
        self.directional = DirectionalLoss()
    
    def forward(self, predictions, targets):
        huber_loss = self.huber(predictions, targets)
        
        if predictions.shape[1] > 1:
            direction_loss = self.directional(predictions, targets)
            total_loss = (1 - self.trend_weight) * huber_loss + self.trend_weight * direction_loss
        else:
            total_loss = huber_loss
        
        return total_loss


class QuantileLoss(nn.Module):
    """
    Quantile loss for probabilistic predictions
    Useful for risk-aware trading
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, predictions, targets):
        losses = []
        
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions
            losses.append(torch.max((q - 1) * errors, q * errors).mean())
        
        return torch.mean(torch.stack(losses))

