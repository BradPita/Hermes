# Usage Guide

## Basic Usage

### Training

```bash
# Train with default settings (Transformer, 100 epochs)
python main.py --mode train

# Train LSTM model
python main.py --mode train --model lstm

# Custom epochs and batch size
python main.py --mode train --epochs 50 --batch_size 64

# Custom learning rate
python main.py --mode train --lr 0.0001
```

### Inference

```bash
# Run inference with best model
python main.py --mode inference

# Use specific checkpoint
python main.py --mode inference --checkpoint financial_model/checkpoints/checkpoint_epoch_50.pt

# Save predictions
python main.py --mode inference --output predictions.csv
```

## Advanced Usage

### Custom Data

Prepare your CSV file with columns: `date`, `open`, `high`, `low`, `close`, `volume`

```bash
python main.py --mode train --data_path data/my_stock_data.csv
```

### Python API

```python
import sys
sys.path.insert(0, 'financial_model')

from src.config import Config
from src.train import Trainer
from src.inference import Predictor
from src.dataset import create_dataloaders
import pandas as pd

# Load data
data = pd.read_csv('data/stock_data.csv')

# Configure
config = Config()
config.model.model_type = 'transformer'
config.training.num_epochs = 50
config.training.batch_size = 32

# Create dataloaders
train_loader, val_loader, test_loader, scaler = create_dataloaders(data, config)

# Train
trainer = Trainer(config)
trainer.train(train_loader, val_loader)

# Inference
predictor = Predictor('financial_model/checkpoints/best_model.pt')
predictions = predictor.predict_from_dataframe(data)
```

### Custom Model Configuration

Edit `financial_model/src/config.py`:

```python
@dataclass
class ModelConfig:
    model_type: str = "transformer"  # or "lstm"
    input_dim: int = 5               # number of features
    hidden_dim: int = 256            # hidden layer size
    num_layers: int = 4              # number of layers
    num_heads: int = 8               # attention heads (transformer only)
    dropout: float = 0.1             # dropout rate
    seq_length: int = 60             # input sequence length
    pred_length: int = 1             # prediction horizon
```

## Tips

1. **GPU Memory**: Reduce `batch_size` if out of memory
2. **Training Speed**: Increase `batch_size` for faster training
3. **Overfitting**: Increase `dropout` or reduce `num_layers`
4. **Underfitting**: Increase `hidden_dim` or `num_layers`
5. **Long Sequences**: Increase `seq_length` for longer patterns

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python main.py --mode train --batch_size 16
```

### Slow Training
```bash
# Increase batch size (if memory allows)
python main.py --mode train --batch_size 128
```

### Poor Performance
- Try different model architectures (transformer vs lstm)
- Adjust sequence length
- Tune learning rate
- Add more training data

