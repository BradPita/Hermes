# Financial Time Series Model

A PyTorch-based deep learning framework for financial time series forecasting with support for Transformer and LSTM architectures.

## Features

- **Multiple Model Architectures**
  - Transformer with Flash Attention 2.8.2 support
  - LSTM for sequential modeling
  
- **GPU Acceleration**
  - CUDA 12.8 support
  - Mixed precision training (AMP)
  - Optimized for NVIDIA RTX 5090

- **Flexible Configuration**
  - Easy-to-use configuration system
  - Customizable model parameters
  - Support for various data formats

- **Production Ready**
  - Model checkpointing
  - Early stopping
  - Inference pipeline

## Project Structure

```
financial_model/
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── model.py           # Model architectures
│   ├── dataset.py         # Data processing
│   ├── train.py           # Training logic
│   └── inference.py       # Inference pipeline
├── data/                  # Data directory
├── checkpoints/           # Model checkpoints
├── logs/                  # Training logs
└── README.md             # This file
```

## Requirements

- Python 3.12
- PyTorch 2.9.1+cu128
- Flash Attention 2.8.2
- CUDA 12.8
- NVIDIA GPU (RTX 5090 recommended)

## Installation

All dependencies are already installed in the virtual environment:

```bash
.\venv\Scripts\activate
```

## Quick Start

### 1. Train a Model

Train a Transformer model:
```bash
python main.py --mode train
```

Train an LSTM model:
```bash
python main.py --mode train --model lstm
```

Custom training parameters:
```bash
python main.py --mode train --epochs 50 --batch_size 64 --lr 0.0001
```

### 2. Run Inference

```bash
python main.py --mode inference
```

With custom checkpoint:
```bash
python main.py --mode inference --checkpoint financial_model/checkpoints/best_model.pt
```

Save predictions to file:
```bash
python main.py --mode inference --output predictions.csv
```

### 3. Use Your Own Data

```bash
python main.py --mode train --data_path your_data.csv
```

Data format (CSV):
```
date,open,high,low,close,volume
2020-01-01,100.0,102.0,99.0,101.0,1000000
2020-01-02,101.0,103.0,100.0,102.0,1100000
...
```

## Configuration

Edit `src/config.py` to customize:

- Model architecture (Transformer/LSTM)
- Hidden dimensions and layers
- Sequence length
- Training hyperparameters
- Data preprocessing options

## Model Architectures

### Transformer
- Multi-head attention mechanism
- Positional encoding
- Flash Attention 2.8.2 optimization
- ~3.1M parameters (default config)

### LSTM
- Bidirectional LSTM layers
- Dropout regularization
- ~1.8M parameters (default config)

## Training Features

- **Mixed Precision Training**: Faster training with AMP
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Automatic training termination
- **Checkpointing**: Save best models automatically
- **Progress Tracking**: Real-time training metrics

## Performance

On NVIDIA RTX 5090:
- Training speed: ~60 batches/second
- Inference: Real-time prediction
- Memory usage: ~2GB VRAM (default config)

## Author

eddy

## License

MIT License

