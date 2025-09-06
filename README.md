# StoxLSTM for Kraken BTC-USD 1-Minute OHLCV

LLM-slopping my way through this paper - https://arxiv.org/pdf/2509.01187. Don't take this repo seriously.

A modular implementation of StoxLSTM for probabilistic time-series forecasting on Kraken BTC-USD 1-minute OHLCV data with up to 48-hour forecasting capability.

## 🏗️ Architecture

The code is organized into modular components:

- **`config.py`** - Configuration management
- **`data_utils.py`** - Data loading, preprocessing, normalization, and dataset creation
- **`model.py`** - StoxLSTM model implementation
- **`train.py`** - Training script
- **`predict.py`** - Inference script for predictions from timestamps

## 🚀 Quick Start

### 1. Training

Train the model with default settings:

```bash
python train.py
```

Train with custom parameters:

```bash
python train.py --epochs 10 --batch_size 32 --lr 1e-4 --beta_kl 10.0
```

### 2. Prediction

Generate a 48-hour forecast from a specific timestamp:

```bash
python predict.py checkpoints/model_epoch_5.pth "2023-12-01 12:00:00"
```

Generate a custom horizon forecast:

```bash
python predict.py checkpoints/model_epoch_5.pth "2023-12-01 12:00:00" --horizon 1440 --save_plot forecast.png --save_csv forecast.csv
```

## 📊 Features

- **48-hour forecasting**: Predict up to 2880 minutes (48 hours) into the future
- **OHLCV support**: Handles Open, High, Low, Close, Volume data
- **Rolling normalization**: 30-day rolling window with no data leakage
- **Temporal splits**: Proper train/val/test splits maintaining temporal order
- **TensorBoard logging**: Comprehensive training metrics and visualizations
- **Checkpoint management**: Save and load model states
- **Flexible inference**: Predict from any timestamp in your data
- **Actual vs Forecast comparison**: Automatically overlays actual data when available for performance evaluation
- **Performance metrics**: Calculates MAE and percentage errors for forecast accuracy

## 🔧 Configuration

Key parameters in `config.py`:

```python
# Windows
L: int = 1440  # lookback (24h)
T: int = 2880  # horizon (48h)
stride_windows: int = 60  # shift between training windows (1h)

# Model
d_model: int = 64
d_latent: int = 16
dropout: float = 0.05

# Training
batch_size: int = 64
epochs: int = 5
lr: float = 3e-4
beta_kl: float = 5.0  # KL weight
```

## 📈 Data Requirements

Your CSV file should contain:
- `timestamp`: epoch seconds or ISO8601 string (UTC assumed)
- `open`, `high`, `low`, `close`, `volume`: OHLCV data

Example:
```csv
timestamp,open,high,low,close,volume
1640995200,47000.0,47500.0,46800.0,47200.0,123.45
1640995260,47200.0,47300.0,47100.0,47250.0,98.76
```

## 🎯 Usage Examples

### Training Examples

```bash
# Basic training
python train.py

# Custom training with different parameters
python train.py --epochs 20 --batch_size 128 --lr 1e-3 --beta_kl 2.0

# Train on different data
python train.py --csv_path data/ETHUSD_1.csv --epochs 10
```

### Prediction Examples

```bash
# 48-hour forecast from specific timestamp
python predict.py checkpoints/model_epoch_5.pth "2023-12-01 12:00:00"

# 24-hour forecast with custom output
python predict.py checkpoints/model_epoch_5.pth "2023-12-01 12:00:00" --horizon 1440 --save_plot my_forecast.png

# 12-hour forecast with CSV output
python predict.py checkpoints/model_epoch_5.pth "2023-12-01 12:00:00" --horizon 720 --save_csv forecast_12h.csv

# Use different checkpoint
python predict.py checkpoints/model_epoch_10.pth "2023-11-15 08:30:00" --horizon 2880

# Forecast with actual data comparison (when available)
python predict.py checkpoints/model_epoch_5.pth "2023-12-15 12:00:00" --horizon 1440 --save_plot comparison.png
```

## 📁 Output Files

### Training Outputs
- **Checkpoints**: `checkpoints/model_epoch_X.pth`
- **TensorBoard logs**: `runs/StoxLSTM_SEED_TIMESTAMP/`
- **Forecast plots**: Generated after each epoch

### Prediction Outputs
- **Plots**: OHLCV charts with historical context, forecast, and actual data comparison
- **CSV files**: Forecast data in OHLCV format
- **Console output**: Forecast summary statistics and accuracy metrics
- **Performance metrics**: MAE, percentage error, and price range comparisons

## 🔍 Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

Key metrics tracked:
- Total loss (ELBO)
- Reconstruction loss
- KL divergence
- Training/validation curves
- GPU memory usage
- Learning rate

## 📈 Visualization Features

### Forecast Plots
The prediction script generates comprehensive OHLCV charts that include:

- **Historical data**: Last 24 hours of actual price data (solid lines)
- **Forecast data**: Predicted future values (dashed lines)
- **Actual data**: Real values for the forecast period when available (solid lines, same colors as forecast)
- **Volume analysis**: Historical, forecast, and actual volume data

### Performance Evaluation
When actual data is available for the forecast period, the system automatically:

- **Overlays actual data** on the same plot for visual comparison
- **Calculates accuracy metrics**:
  - Mean Absolute Error (MAE) for close prices
  - Mean Percentage Error
  - Price range comparisons
  - Final price accuracy
- **Provides detailed statistics** in console output

### Color Coding
- **Blue**: Open prices
- **Green**: High prices  
- **Red**: Low prices
- **Black**: Close prices (thicker line for emphasis)
- **Orange**: Volume data
- **Dashed lines**: Forecasts
- **Solid lines**: Historical and actual data

## ⚙️ Advanced Usage

### Custom Configuration

Modify `config.py` for your specific needs:

```python
# For longer forecasting
cfg.T = 4320  # 72 hours

# For different model size
cfg.d_model = 128
cfg.d_latent = 32

# For different data
cfg.csv_path = "data/your_data.csv"
cfg.cols = ("open", "high", "low", "close", "volume")
```

### Programmatic Usage

```python
from config import cfg
from data_utils import load_kraken_ohlc, create_simple_temporal_split
from model import create_model
from train import train_loop

# Load data
df = load_kraken_ohlc("data/BTCUSD_1.csv")

# Create model
model = create_model()

# Train
model = train_loop(model, train_loader, val_loader, cfg)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `d_model`
2. **Empty validation set**: Check your data size and split ratios
3. **NaN losses**: Adjust `beta_kl` or check data quality
4. **Slow training**: Ensure CUDA is available and reduce `stride_windows`

### Performance Tips

- Use CUDA for faster training
- Adjust `stride_windows` for more/fewer training samples
- Monitor GPU memory usage in TensorBoard
- Use `drop_last=False` for debugging dataset issues

## 📊 Model Architecture

The StoxLSTM model consists of:

1. **Patch Embedding**: Converts time series to patches
2. **Stochastic Cell**: GRU with latent variables
3. **Posterior Network**: Bidirectional LSTM for inference
4. **Multi-channel**: Separate processing for each OHLCV channel

## 🎯 Forecasting Capabilities

- **Horizon**: Up to 48 hours (2880 minutes)
- **Resolution**: 1-minute intervals
- **Channels**: Open, High, Low, Close, Volume
- **Uncertainty**: Probabilistic forecasts with latent variables
- **Temporal**: Maintains temporal relationships

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This implementation is provided as-is for educational and research purposes.
