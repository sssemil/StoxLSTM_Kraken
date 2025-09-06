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
import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from datetime import datetime

from config import cfg
from data_utils import (
    load_kraken_ohlc, resample_and_fill_1m, compute_rolling_norm, to_patches
)
from model import create_model
from plot_utils import plot_ohlcv_forecast, get_actual_data_for_period, calculate_forecast_metrics, print_forecast_metrics, open_plot
from forecast_utils import forecast_from_timestamp

def load_checkpoint(checkpoint_path, device=None):
    """Load model from checkpoint."""
    device = device or cfg.device
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load with weights_only=False to allow config objects
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)  # allow raw state_dict
    model = create_model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"State dict loaded with missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    loaded_cfg = checkpoint.get('config', cfg)
    return model, loaded_cfg

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
        ts = ts.tz_localize(df.index.tz or 'UTC')
    else:
        ts = ts.tz_convert(df.index.tz or 'UTC')
    
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
        delta = abs((actual_ts - ts).total_seconds())
        print(f"Timestamp {ts} not found. Using closest: {actual_ts} (Δ={delta:.0f}s)")
        return closest_idx


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
    
    # Update config with command line arguments
    cfg.csv_path = args.csv_path
    cfg.device = args.device
    
    print("🔮 StoxLSTM Prediction")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Timestamp: {args.timestamp}")
    print(f"Horizon: {args.horizon} minutes ({args.horizon/60:.1f} hours)")
    print(f"CSV path: {cfg.csv_path}")
    print(f"Device: {cfg.device}")
    print("=" * 50)
    
    # Load model
    model, model_cfg = load_checkpoint(args.checkpoint, cfg.device)
    
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
    fut_df, hist_start, hist_end = forecast_from_timestamp(
        model, norm, mean, std, cfg.cols, args.timestamp,
        model_cfg.L, args.horizon, model_cfg.P, model_cfg.S, cfg.device
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
