#!/usr/bin/env python3
"""
Prediction script for StoxLSTM model.
Run inference from a specific timestamp and generate forecasts.
"""
import argparse
import os
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import cfg
from data_utils import (
    load_kraken_ohlc, resample_and_fill_1m, compute_rolling_norm, to_patches
)
from model import create_model
from plot_utils import plot_ohlcv_forecast, get_actual_data_for_period, calculate_forecast_metrics, print_forecast_metrics, open_plot

def load_checkpoint(checkpoint_path, device=None):
    """Load model from checkpoint."""
    if device is None:
        device = cfg.device
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Trying to load only the model state dict...")
        # Try to load just the state dict
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Checkpoint does not contain 'model_state_dict'")
    
    # Create model
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load config if available
    if 'config' in checkpoint:
        try:
            loaded_cfg = checkpoint['config']
            print(f"Loaded config: L={loaded_cfg.L}, T={loaded_cfg.T}, P={loaded_cfg.P}, S={loaded_cfg.S}")
            return model, loaded_cfg
        except Exception as e:
            print(f"Could not load config from checkpoint: {e}")
            print("Using current config")
            return model, cfg
    else:
        print("No config in checkpoint, using current config")
        return model, cfg

def find_timestamp_index(df, timestamp):
    """Find the index of a timestamp in the dataframe."""
    if isinstance(timestamp, str):
        # Parse timestamp string
        try:
            # Try different formats
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
                try:
                    ts = pd.to_datetime(timestamp, format=fmt)
                    break
                except:
                    continue
            else:
                ts = pd.to_datetime(timestamp)
        except:
            raise ValueError(f"Could not parse timestamp: {timestamp}")
    else:
        ts = timestamp
    
    # Make timezone aware if needed
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    
    # Find closest index
    try:
        idx = df.index.get_loc(ts)
        return idx
    except KeyError:
        # Find closest timestamp
        closest_idx = df.index.searchsorted(ts)
        if closest_idx >= len(df):
            closest_idx = len(df) - 1
        elif closest_idx > 0:
            # Check which is closer
            if abs((df.index[closest_idx] - ts).total_seconds()) > abs((df.index[closest_idx-1] - ts).total_seconds()):
                closest_idx = closest_idx - 1
        
        actual_ts = df.index[closest_idx]
        print(f"Timestamp {ts} not found. Using closest: {actual_ts}")
        return closest_idx

@torch.no_grad()
def predict_from_timestamp(model, df_norm, df_mean, df_std, cols, timestamp, 
                          L, T, P, S, device):
    """Generate prediction from a specific timestamp."""
    # Find the timestamp index
    start_idx = find_timestamp_index(df_norm, timestamp)
    
    # Ensure we have enough history
    if start_idx < L:
        raise ValueError(f"Not enough history. Need at least {L} samples before timestamp. "
                        f"Available: {start_idx}")
    
    # Get the lookback window
    hist_start = start_idx - L
    hist_end = start_idx
    
    print(f"Using history from {df_norm.index[hist_start]} to {df_norm.index[hist_end-1]}")
    print(f"Predicting from {df_norm.index[start_idx]} for {T} minutes")
    
    # Prepare input data
    C = len(cols)
    x = torch.tensor(df_norm[list(cols)].iloc[hist_start:hist_end].values, 
                     dtype=torch.float32).unsqueeze(0)  # [1, L, C]
    
    # Convert to patches
    patches_hist_list = []
    for c in range(C):
        ph = to_patches(x[:, :, c].unsqueeze(-1), P, S)  # [1, Np_hist, P, 1]
        patches_hist_list.append(ph.to(device))
    
    # Generate forecast
    Np_future = math.ceil(T / S)
    ypatch_list = model.forecast(patches_hist_list, Np_future)
    
    # Convert patches back to time series
    fut_list = []
    for arr in ypatch_list:
        hist_len = arr.size(1) - Np_future
        fut = arr[:, hist_len:, :] if arr.ndim == 3 else arr[:, hist_len:]
        fut_list.append(fut.squeeze(0).cpu().numpy())  # [Np_future]
    
    # Interpolate to minute-level
    fut_minute = []
    for f in fut_list:
        # Instead of simple repetition, use linear interpolation between patch values
        # This creates smoother transitions and reduces straight line artifacts
        if len(f) > 1:
            # Create interpolation points
            patch_indices = np.arange(len(f)) * S
            minute_indices = np.arange(T)
            # Interpolate between patch values
            rep = np.interp(minute_indices, patch_indices, f)
        else:
            # Fallback to simple repetition if only one patch
            rep = np.repeat(f, S)[:T]
        fut_minute.append(rep)
    fut_minute = np.stack(fut_minute, axis=-1)  # [T, C] (normalized)
    
    # Denormalize
    mu_tail = df_mean[list(cols)].iloc[start_idx-1:start_idx].values  # [1, C]
    sd_tail = df_std[list(cols)].iloc[start_idx-1:start_idx].values  # [1, C]
    fut_denorm = fut_minute * (sd_tail + cfg.epsilon) + mu_tail  # [T, C]
    
    # Create forecast dataframe
    start_ts = df_norm.index[start_idx]
    idx = pd.date_range(start_ts, periods=T, freq="1min", tz=df_norm.index.tz)
    fut_df = pd.DataFrame(fut_denorm, index=idx, columns=cols)
    
    return fut_df, hist_start, hist_end

def plot_prediction(df, fut_df, hist_start, hist_end, timestamp, save_path=None):
    """Plot the prediction with historical context and actual data for comparison."""
    # Get historical data for context (last 24 hours)
    hist_data = df.iloc[hist_start:hist_end].copy()
    
    # Get actual data for the forecast period (if available)
    forecast_start = fut_df.index[0]
    forecast_end = fut_df.index[-1]
    actual_data = get_actual_data_for_period(df, forecast_start, forecast_end)
    
    # Create title
    title = f'BTC-USD Price Forecast from {timestamp}'
    if actual_data is not None:
        title += ' (with Actual Data)'
    
    # Create plot using shared utility
    plot_path = plot_ohlcv_forecast(hist_data, fut_df, actual_data, title=title, save_path=save_path, show_volume=True)
    
    # Open the plot
    open_plot(plot_path)
    
    # Print comparison statistics if actual data is available
    if actual_data is not None and len(actual_data) > 0:
        print(f"\n📊 Forecast vs Actual Comparison:")
        print(f"  Forecast period: {forecast_start} to {forecast_end}")
        print(f"  Actual period: {actual_data.index[0]} to {actual_data.index[-1]}")
        
        # Calculate and print metrics using shared utility
        metrics = calculate_forecast_metrics(fut_df, actual_data)
        print_forecast_metrics(metrics)

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Run StoxLSTM prediction from timestamp')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('timestamp', type=str, help='Timestamp to predict from (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--csv_path', type=str, default=cfg.csv_path, help='Path to CSV file')
    parser.add_argument('--horizon', type=int, default=2880, help='Forecast horizon in minutes (default: 2880 = 48h)')
    parser.add_argument('--save_plot', type=str, help='Path to save plot')
    parser.add_argument('--save_csv', type=str, help='Path to save forecast CSV')
    parser.add_argument('--device', type=str, default=cfg.device, help='Device to use')
    
    args = parser.parse_args()
    
    print("🔮 StoxLSTM Prediction")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Timestamp: {args.timestamp}")
    print(f"Horizon: {args.horizon} minutes ({args.horizon/60:.1f} hours)")
    print(f"CSV path: {args.csv_path}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Load model
    model, model_cfg = load_checkpoint(args.checkpoint, args.device)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df_raw = load_kraken_ohlc(args.csv_path, cfg.time_col, cfg.cols, tz=cfg.tz)
    
    if cfg.resample_1m:
        df = resample_and_fill_1m(df_raw)
    else:
        df = df_raw.copy()
    
    # Rolling normalization
    norm, mean, std = compute_rolling_norm(
        df, window=cfg.roll_days * 24 * 60, min_periods=cfg.min_periods, eps=cfg.epsilon
    )
    valid = norm.dropna().index
    df = df.loc[valid]
    norm = norm.loc[valid]
    mean = mean.loc[valid]
    std = std.loc[valid]
    
    print(f"Data loaded: {len(df)} samples from {df.index[0]} to {df.index[-1]}")
    
    # Generate prediction
    print("🔮 Generating prediction...")
    fut_df, hist_start, hist_end = predict_from_timestamp(
        model, norm, mean, std, cfg.cols, args.timestamp,
        model_cfg.L, args.horizon, model_cfg.P, model_cfg.S, args.device
    )
    
    # Print forecast summary
    print(f"\n📈 Forecast Summary:")
    print(f"  Start: {fut_df.index[0]}")
    print(f"  End: {fut_df.index[-1]}")
    print(f"  Duration: {args.horizon} minutes ({args.horizon/60:.1f} hours)")
    print(f"  Close price range: ${fut_df['close'].min():.2f} - ${fut_df['close'].max():.2f}")
    print(f"  Final close: ${fut_df['close'].iloc[-1]:.2f}")
    
    # Save forecast to CSV if requested
    if args.save_csv:
        fut_df.to_csv(args.save_csv)
        print(f"Forecast saved to: {args.save_csv}")
    
    # Plot prediction
    print("📊 Creating plot...")
    plot_prediction(df, fut_df, hist_start, hist_end, args.timestamp, args.save_plot)
    
    print("✅ Prediction completed!")

if __name__ == "__main__":
    main()
