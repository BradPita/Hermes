# A股多维度金融模型 / A-Share Multi-Dimensional Financial Model

完整的金融时间序列预测系统，支持多维度条件化训练
A complete financial time series prediction system with multi-dimensional conditional training for A-share stocks.

## 作者 / Author
eddy

## 版本 / Version
2.0 (Updated: 2025-11-14)

## 核心特性 / Features

- **🌐 双语界面 / Bilingual Interface**: 中英文双语支持 / Chinese & English support
- **📥 集成数据下载 / Integrated Data Download**: Gradio界面直接下载A股数据 / Download A-share data directly in Gradio
- **🎨 多维度训练 / Multi-Dimensional Training**: 行业、风格、市场环境感知 / Industry, style factors, and market regime awareness
- **⚡ Flash Attention 2.8.2**: RTX 5090 GPU加速训练 / GPU-accelerated training on RTX 5090
- **📊 真实A股数据 / Real A-Share Data**: 1999年至今历史数据 (3500+股票) / Historical data from 1999 to present (3500+ stocks)
- **🖥️ Web界面 / Web Interface**: 基于Gradio的训练和推理面板 / Gradio-based training and inference panel
- **🧠 条件化模型 / Conditional Model**: 共享骨干网络+上下文感知预测 / Shared backbone with context-aware predictions

## 快速开始 / Quick Start

### 方法1: 使用启动脚本 / Method 1: Use Start Script

**Windows:**
```bash
start.bat
```

### 方法2: 手动启动 / Method 2: Manual Start

```bash
python gradio_app.py
```

浏览器访问 / Open browser: http://127.0.0.1:7860

## 完整使用流程 / Complete Workflow

### 步骤1: 下载数据 / Step 1: Download Data

1. 打开Gradio界面 / Open Gradio interface
2. 进入 "📥 数据下载 / Data Download" 标签页 / Go to "Data Download" tab
3. 设置下载股票数 (建议先100只测试) / Set number of stocks (recommend 100 for testing)
4. 设置起始日期 (默认1999-01-01) / Set start date (default: 1999-01-01)
5. 点击 "🚀 开始下载 / Start Download"
6. 等待下载完成 / Wait for completion
7. 点击 "🔄 刷新状态 / Refresh Status" 验证 / Click to verify

**数据将保存到 / Data will be saved to:** `full_stock_data/training_data/`

### 步骤2: 分析股票池 (可选) / Step 2: Analyze Universe (Optional)

1. 进入 "🎨 多维度训练 / Multi-Dimensional Training" 标签页
2. 设置 "分析股票数 / Max Stocks to Analyze"
3. 点击 "📊 分析股票池 / Analyze Universe"
4. 查看行业分布和风格因子 / Review industry distribution and style factors

### 步骤3: 训练模型 / Step 3: Train Model

**选项A: 基础训练 (快速测试) / Option A: Basic Training (Quick Test)**
1. 进入 "🎯 基础训练 / Basic Training" 标签页
2. 选择模型类型 (推荐Transformer) / Select model type (Transformer recommended)
3. 配置参数 / Configure parameters
4. 点击 "🚀 开始训练 / Start Training"

**选项B: 多维度训练 (推荐) / Option B: Multi-Dimensional Training (Recommended)**
1. 进入 "🎨 多维度训练 / Multi-Dimensional Training" 标签页
2. 选择 "Unified Conditional Model"
3. 设置最大股票数 (100-500) / Set max stocks (100-500)
4. 配置批次大小、轮数、学习率 / Configure batch size, epochs, learning rate
5. 点击 "🚀 开始多维度训练 / Start Multi-Dim Training"

### 步骤4: 模型推理 / Step 4: Model Inference

1. 进入 "🔮 Inference" 标签页
2. 选择检查点路径 / Select checkpoint path
3. 设置测试样本数 / Set test samples
4. 点击 "🔮 Run Inference"
5. 查看预测结果 / Review predictions

## Project Structure

```
fork/
├── gradio_app.py                    # Main Gradio web interface
├── batch_download.py                # A-share data downloader
├── financial_model/                 # Core model implementation
│   ├── src/
│   │   ├── config.py               # Configuration
│   │   ├── model.py                # Base models (Transformer/LSTM)
│   │   ├── conditional_model.py    # Multi-dimensional conditional model
│   │   ├── dataset.py              # Basic dataset
│   │   ├── multi_dim_dataset.py    # Multi-dimensional dataset
│   │   ├── stock_metadata.py       # Industry & style classification
│   │   ├── market_regime.py        # Market regime detection
│   │   ├── train.py                # Trainer
│   │   └── inference.py            # Predictor
│   └── checkpoints/                # Saved models
├── full_stock_data/                # Downloaded stock data
│   ├── training_data/              # CSV files (3500+ stocks)
│   └── metadata.json               # Download progress
└── old/                            # Archived files

```

## Multi-Dimensional Training

### Three Core Dimensions

1. **Industry Classification** (11 categories)
   - finance, consumer, technology, healthcare, industrial
   - materials, energy, utilities, real_estate, telecom, other

2. **Style Factors** (5 categories)
   - Market Cap: mega/large/mid/small/micro
   - Value/Growth: deep_value/value/balanced/growth
   - Volatility: low/medium/high
   - Momentum: strong/positive/neutral/negative/reversal

3. **Market Regime** (4 states)
   - bull, bear, sideways, volatile

### Model Architecture

```
Input (OHLCV) → CNN Backbone → LSTM Backbone → Backbone Features
                                                        ↓
Industry Embedding (32 dim) ────────────────────────→ Fusion Layer → Prediction Head
Style Embedding (16 dim) ───────────────────────────→
Regime Embedding (16 dim) ──────────────────────────→
```

### Training Modes

1. **Unified Conditional Model** (RECOMMENDED)
   - One model learns all industries with conditional inputs
   - Best performance and efficiency
   - Model size: ~50MB

2. **Single Industry Model**
   - Train on specific industry for specialized predictions
   - Useful for industry-specific strategies

## Data Format

CSV files with columns:
- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Performance

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **Training Speed**: ~88 it/s
- **Model Parameters**: 2.5M
- **Training Time**: ~3-4 hours (100 stocks, 20 epochs)
- **Data Size**: 3500+ stocks, ~2.4GB

## Key Files

### Main Scripts
- `gradio_app.py` - Web interface with 4 tabs (Training, Inference, Custom Data, Multi-Dim Training)
- `batch_download.py` - Download A-share data from TDX server

### Model Files
- `financial_model/src/conditional_model.py` - Multi-dimensional conditional model
- `financial_model/src/multi_dim_dataset.py` - Dataset with automatic labeling
- `financial_model/src/stock_metadata.py` - Industry and style classification
- `financial_model/src/market_regime.py` - Market regime detection

## Requirements

- Python 3.12.8
- PyTorch 2.9.1+cu128
- Flash Attention 2.8.2
- Gradio 5.7.1
- pytdx 1.72
- pandas, numpy, plotly

## Usage Examples

### Analyze Stock Universe

```python
# In Gradio interface:
# 1. Go to "Multi-Dimensional Training" tab
# 2. Set "Max Stocks to Analyze" = 100
# 3. Click "Analyze Universe"
# 
# Output:
# - Industry distribution
# - Volatility styles
# - Current market regime
```

### Train Unified Model

```python
# In Gradio interface:
# 1. Select "Unified Conditional Model"
# 2. Set Max Stocks = 100
# 3. Set Batch Size = 64
# 4. Set Epochs = 20
# 5. Click "Start Multi-Dim Training"
#
# Model saved to: financial_model/checkpoints/best_multi_dim_model.pt
```

## Tips

- Start with 100 stocks for testing
- Use batch size 64 for RTX 5090
- Unified Conditional Model is recommended for best results
- Monitor training loss - should decrease to < 0.01
- Check validation loss to avoid overfitting

## Next Steps

1. Wait for batch download to complete (3500+ stocks)
2. Analyze full universe
3. Train on 500+ stocks for production model
4. Evaluate performance by industry and market regime
5. Fine-tune parameters based on results

## License

MIT

