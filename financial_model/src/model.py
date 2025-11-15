"""
Financial Time Series Models

Author: eddy
"""

import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class FinancialTransformer(nn.Module):
    """Transformer model for financial time series with Flash Attention support"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
        pred_length: int = 1,
        use_flash_attention: bool = True
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        self.use_flash_attention = use_flash_attention

        if use_flash_attention and torch.cuda.is_available():
            try:
                import flash_attn
                print(f"Using Flash Attention {flash_attn.__version__} for acceleration!")

                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(False)
            except ImportError:
                print("Flash Attention not available, using standard attention")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(hidden_dim, pred_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        x = self.transformer_encoder(x)
        
        x = x[:, -1, :]
        
        output = self.output_projection(x)
        
        return output

class FinancialLSTM(nn.Module):
    """LSTM model for financial time series"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        pred_length: int = 1
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, pred_length)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        
        last_output = lstm_out[:, -1, :]
        
        output = self.fc(last_output)
        
        return output

def create_model(config) -> nn.Module:
    """Create model based on configuration"""
    
    if config.model.model_type == "transformer":
        model = FinancialTransformer(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            pred_length=config.model.pred_length,
            use_flash_attention=config.model.use_flash_attention
        )
    elif config.model.model_type == "lstm":
        model = FinancialLSTM(
            input_dim=config.model.input_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            pred_length=config.model.pred_length
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.model_type}")
    
    return model.to(config.model.device)

