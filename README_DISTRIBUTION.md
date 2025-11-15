# Stock Prediction System

Multi-dimensional conditional stock prediction system with Gradio web interface.

## Features

- Multi-dimensional conditional model (Industry + Style + Market Regime)
- Real-time training visualization
- GPU-accelerated data preprocessing
- Multiple loss functions (MSE, Trend Aware, Huber, Directional)
- Single stock and multi-stock training modes
- Interactive prediction interface

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: RTX 3090/4090/5090)
- 16GB+ RAM
- 50GB+ disk space for data

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd stock_prediction_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download Stock Data

```bash
python gradio_app.py
```

Navigate to "Data Download" tab and download 100-500 stocks.

### 2. Train Model

**Recommended Parameters (100 stocks):**
- Training Mode: Unified Conditional Model
- Max Stocks: 100
- Batch Size: 64
- Epochs: 30
- Learning Rate: 0.001
- Loss Function: Trend Aware
- Mixed Precision: Enabled

### 3. Predict

Use "Single Stock Prediction" tab to predict future prices.

## Project Structure

```
stock_prediction_system/
├── gradio_app.py                    # Main Gradio interface
├── financial_model/
│   ├── src/
│   │   ├── config.py               # Configuration
│   │   ├── model.py                # Base model
│   │   ├── conditional_model.py    # Multi-dim model
│   │   ├── train.py                # Training logic
│   │   ├── improved_loss.py        # Custom loss functions
│   │   ├── multi_dim_dataset.py    # Dataset with GPU acceleration
│   │   ├── stock_metadata.py       # Industry/Style classification
│   │   ├── market_regime.py        # Market regime detection
│   │   └── conditional_inference.py # Prediction logic
│   └── checkpoints/
│       └── conditional_best_model.pt # Pre-trained model
├── requirements.txt
└── README_DISTRIBUTION.md

```

## Loss Functions

- **MSE**: Standard mean squared error
- **Trend Aware** (Recommended): Encourages trend prediction
- **Huber Trend**: Robust to outliers
- **Directional**: Penalizes wrong direction

## Hardware Recommendations

- **GPU**: RTX 3090 (24GB) or better
- **RAM**: 32GB+
- **Storage**: SSD with 100GB+ free space

## Training Tips

1. Start with 100 stocks for testing
2. Use Trend Aware loss function to avoid flat predictions
3. Monitor validation loss curve
4. Save models every 5 epochs
5. Use mixed precision (AMP) for faster training

## Troubleshooting

**Flat predictions?**
- Use Trend Aware loss function
- Train for at least 30 epochs
- Use 100+ stocks

**Out of memory?**
- Reduce batch size to 32
- Reduce max stocks to 50
- Enable mixed precision

**Slow training?**
- Enable GPU acceleration
- Use mixed precision (AMP)
- Increase batch size to 128

## Author

eddy

## License

MIT

