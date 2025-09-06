#!/usr/bin/env python3
"""
Training script for StoxLSTM model.
"""
import os
import time
import math
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from data_utils import (
    seed_all, load_kraken_ohlc, resample_and_fill_1m, compute_rolling_norm,
    create_simple_temporal_split, create_datasets_and_loaders, batch_to_patches
)
from model import create_model, compute_elbo
from plot_utils import plot_ohlcv_forecast, get_actual_data_for_period, calculate_forecast_metrics, print_forecast_metrics, open_plot

def save_checkpoint_and_plot(model, df, norm, mean, std, epoch, is_random=False):
    """Save model checkpoint and create forecast plot."""
    import os
    
    # Create checkpoints directory
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Save model checkpoint
    if is_random:
        checkpoint_path = f"{cfg.checkpoint_dir}/model_random_weights.pth"
        plot_title = "Random Weights Model"
    else:
        checkpoint_path = f"{cfg.checkpoint_dir}/model_epoch_{epoch}.pth"
        plot_title = f"Epoch {epoch} Model"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': cfg
    }, checkpoint_path)
    
    # Create forecast from a timestamp where we have actual data for comparison
    # Use a timestamp that's far enough from the end to have actual data
    forecast_timestamp = df.index[-cfg.T - 1000]  # Go back enough to have actual data
    fut_df, hist_start, hist_end = forecast_from_timestamp(
        model, norm, mean, std, cfg.cols, forecast_timestamp, cfg.L, cfg.T, cfg.P, cfg.S, cfg.device
    )
    
    # Get historical data for context
    hist_data = df.iloc[hist_start:hist_end].copy()
    
    # Get actual data for the forecast period (should always be available now)
    forecast_start = fut_df.index[0]
    forecast_end = fut_df.index[-1]
    actual_data = get_actual_data_for_period(df, forecast_start, forecast_end)
    
    # Create plot using shared utility with actual data for comparison
    title = f'BTC-USD Price Forecast - {plot_title}'
    if actual_data is not None and len(actual_data) > 0:
        title += ' (with Actual Data)'
    
    plot_path = plot_ohlcv_forecast(hist_data, fut_df, actual_data, title=title, show_volume=True)
    
    # Open the plot
    open_plot(plot_path)
    
    # Print forecast metrics if actual data is available
    if actual_data is not None and len(actual_data) > 0:
        print(f"\n📊 Forecast vs Actual Comparison for {plot_title}:")
        print(f"  Forecast period: {forecast_start} to {forecast_end}")
        print(f"  Actual period: {actual_data.index[0]} to {actual_data.index[-1]}")
        
        # Calculate and print metrics
        metrics = calculate_forecast_metrics(fut_df, actual_data)
        print_forecast_metrics(metrics)
    else:
        print(f"  No actual data available for comparison in forecast period")
    
    print(f"✅ Checkpoint saved: {checkpoint_path}")
    return fut_df

@torch.no_grad()
def forecast_future(model, df_norm, df_mean, df_std, cols, L, T, P, S, device):
    """Create forecast for the next T minutes from the end of the dataset."""
    import numpy as np
    import pandas as pd
    from data_utils import to_patches
    
    C = len(cols)
    x = torch.tensor(df_norm[list(cols)].iloc[-(L):].values, dtype=torch.float32).unsqueeze(0)  # [1,L,C]

    patches_hist_list = []
    for c in range(C):
        ph = to_patches(x[:, :, c].unsqueeze(-1), P, S)  # [1,Np_hist,P,1]
        patches_hist_list.append(ph.to(device))

    Np_future = math.ceil(T / S)
    ypatch_list = model.forecast(patches_hist_list, Np_future)  # list of [1, Np_hist+Np_future]

    fut_list = []
    for arr in ypatch_list:
        hist_len = arr.size(1) - Np_future
        fut = arr[:, hist_len:, :] if arr.ndim == 3 else arr[:, hist_len:]
        fut_list.append(fut.squeeze(0).cpu().numpy())  # [Np_future]

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

    mu_tail = df_mean[list(cols)].iloc[-1:].values  # [1,C]
    sd_tail = df_std[list(cols)].iloc[-1:].values  # [1,C]
    fut_denorm = fut_minute * (sd_tail + cfg.epsilon) + mu_tail  # [T,C]

    start_ts = df_norm.index[-1]
    idx = pd.date_range(start_ts + pd.Timedelta(minutes=1), periods=T, freq="1min", tz=df_norm.index.tz)
    fut_df = pd.DataFrame(fut_denorm, index=idx, columns=cols)
    return fut_df

@torch.no_grad()
def forecast_from_timestamp(model, df_norm, df_mean, df_std, cols, timestamp, L, T, P, S, device):
    """Create forecast from a specific timestamp where we have actual data for comparison."""
    import numpy as np
    import pandas as pd
    from data_utils import to_patches
    
    # Find the timestamp index
    if isinstance(timestamp, str):
        ts = pd.to_datetime(timestamp)
    else:
        ts = timestamp
    
    # Make timezone aware if needed
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    
    # Find closest index
    try:
        start_idx = df_norm.index.get_loc(ts)
    except KeyError:
        # Find closest timestamp
        closest_idx = df_norm.index.searchsorted(ts)
        if closest_idx >= len(df_norm):
            closest_idx = len(df_norm) - 1
        elif closest_idx > 0:
            # Check which is closer
            if abs((df_norm.index[closest_idx] - ts).total_seconds()) > abs((df_norm.index[closest_idx-1] - ts).total_seconds()):
                closest_idx = closest_idx - 1
        start_idx = closest_idx
    
    # Ensure we have enough history
    if start_idx < L:
        start_idx = L
    
    # Ensure we have enough future data for comparison
    if start_idx + T >= len(df_norm):
        start_idx = len(df_norm) - T - 1
    
    # Get the lookback window
    hist_start = start_idx - L
    hist_end = start_idx
    
    C = len(cols)
    x = torch.tensor(df_norm[list(cols)].iloc[hist_start:hist_end].values, 
                     dtype=torch.float32).unsqueeze(0)  # [1, L, C]
    
    patches_hist_list = []
    for c in range(C):
        ph = to_patches(x[:, :, c].unsqueeze(-1), P, S)  # [1, Np_hist, P, 1]
        patches_hist_list.append(ph.to(device))
    
    Np_future = math.ceil(T / S)
    ypatch_list = model.forecast(patches_hist_list, Np_future)
    
    fut_list = []
    for arr in ypatch_list:
        hist_len = arr.size(1) - Np_future
        fut = arr[:, hist_len:, :] if arr.ndim == 3 else arr[:, hist_len:]
        fut_list.append(fut.squeeze(0).cpu().numpy())  # [Np_future]
    
    fut_minute = []
    for f in fut_list:
        if len(f) > 1:
            patch_indices = np.arange(len(f)) * S
            minute_indices = np.arange(T)
            rep = np.interp(minute_indices, patch_indices, f)
        else:
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

def train_loop(model, train_loader, val_loader, cfg, df, norm, mean, std):
    """Main training loop."""
    # Create TensorBoard writer
    log_dir = f"{cfg.log_dir}/StoxLSTM_{cfg.seed}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float('inf')
    best_state = None
    
    print(f"Starting training for {cfg.epochs} epochs...")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Beta KL: {cfg.beta_kl}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"To view: tensorboard --logdir={log_dir}")
    print("-" * 60)

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        model.train()
        losses = []
        recon_losses = []
        kl_losses = []
        
        # Training phase
        for batch_idx, batch in enumerate(train_loader):
            ph_list, pa_list, tgt_list = batch_to_patches(batch, cfg.P, cfg.S)
            ph_list = [t.to(cfg.device) for t in ph_list]
            pa_list = [t.to(cfg.device) for t in pa_list]
            tgt_list = [t.to(cfg.device) for t in tgt_list]

            ys, priors, posts, _ = model(ph_list, pa_list)
            loss, rec, kl = compute_elbo(ys, priors, posts, tgt_list, beta_kl=cfg.beta_kl)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            losses.append(loss.item())
            recon_losses.append(rec)
            kl_losses.append(kl)
            
            # Log batch-level metrics to TensorBoard (every 10 batches to avoid clutter)
            if batch_idx % 10 == 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
                writer.add_scalar('Batch/Train_Recon', rec, global_step)
                writer.add_scalar('Batch/Train_KL', kl, global_step)

        # Validation phase
        model.eval()
        with torch.no_grad():
            v_losses = []
            v_recon_losses = []
            v_kl_losses = []
            
            for batch in val_loader:
                ph_list, pa_list, tgt_list = batch_to_patches(batch, cfg.P, cfg.S)
                ph_list = [t.to(cfg.device) for t in ph_list]
                pa_list = [t.to(cfg.device) for t in pa_list]
                tgt_list = [t.to(cfg.device) for t in tgt_list]
                ys, priors, posts, _ = model(ph_list, pa_list)
                loss, rec, kl = compute_elbo(ys, priors, posts, tgt_list, beta_kl=cfg.beta_kl)
                v_losses.append(loss.item())
                v_recon_losses.append(rec)
                v_kl_losses.append(kl)
            
            v_mean = float(torch.tensor(v_losses).mean()) if v_losses else float('nan')
            v_recon_mean = float(torch.tensor(v_recon_losses).mean()) if v_recon_losses else float('nan')
            v_kl_mean = float(torch.tensor(v_kl_losses).mean()) if v_kl_losses else float('nan')

        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Log epoch-level metrics to TensorBoard
        writer.add_scalar('Loss/Total', torch.tensor(losses).mean(), epoch)
        writer.add_scalar('Loss/Total_Val', v_mean, epoch)
        writer.add_scalar('Loss/Reconstruction', torch.tensor(recon_losses).mean(), epoch)
        writer.add_scalar('Loss/Reconstruction_Val', v_recon_mean, epoch)
        writer.add_scalar('Loss/KL_Divergence', torch.tensor(kl_losses).mean(), epoch)
        writer.add_scalar('Loss/KL_Divergence_Val', v_kl_mean, epoch)
        writer.add_scalar('Epoch/Time', epoch_time, epoch)
        
        # Print detailed metrics
        print(f"Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train Loss: {torch.tensor(losses).mean():.4f} (Recon: {torch.tensor(recon_losses).mean():.4f}, KL: {torch.tensor(kl_losses).mean():.4f}) | "
              f"Val Loss: {v_mean:.4f} (Recon: {v_recon_mean:.4f}, KL: {v_kl_mean:.4f}) | "
              f"Time: {epoch_time:.1f}s")
        
        # Update best model
        if v_mean < best_val:
            best_val = v_mean
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            writer.add_scalar('Epoch/Best_Val_Loss', best_val, epoch)
            print(f"  → New best validation loss: {best_val:.4f}")
        
        # Save checkpoint and plot forecast after each epoch
        print(f"  📊 Creating forecast plot for epoch {epoch}...")
        save_checkpoint_and_plot(model, df, norm, mean, std, epoch)
        
        # Log CUDA memory usage if available
        if cfg.device == "cuda" and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, epoch)
            writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, epoch)

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nTraining completed! Best validation loss: {best_val:.4f}")
    
    # Close TensorBoard writer
    writer.close()
    print(f"\nTensorBoard logs saved to: {log_dir}")
    print(f"To view: tensorboard --logdir={log_dir}")
    
    return model

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train StoxLSTM model')
    parser.add_argument('--csv_path', type=str, default=cfg.csv_path, help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=cfg.epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=cfg.batch_size, help='Batch size')
    parser.add_argument('--lr', type=float, default=cfg.lr, help='Learning rate')
    parser.add_argument('--beta_kl', type=float, default=cfg.beta_kl, help='KL weight')
    parser.add_argument('--device', type=str, default=cfg.device, help='Device to use')
    parser.add_argument('--seed', type=int, default=cfg.seed, help='Random seed')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    cfg.csv_path = args.csv_path
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.beta_kl = args.beta_kl
    cfg.device = args.device
    cfg.seed = args.seed
    
    # Set random seed
    seed_all(cfg.seed)
    
    print("🚀 StoxLSTM Training")
    print("=" * 50)
    print(f"CSV path: {cfg.csv_path}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Beta KL: {cfg.beta_kl}")
    print(f"Device: {cfg.device}")
    print(f"Seed: {cfg.seed}")
    print("=" * 50)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df_raw = load_kraken_ohlc(cfg.csv_path, cfg.time_col, cfg.cols, tz=cfg.tz)
    
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
    
    # Create temporal split
    print("📈 Creating temporal train/val/test split...")
    (train_norm, val_norm, test_norm, 
     train_mean, val_mean, test_mean, 
     train_std, val_std, test_std) = create_simple_temporal_split(norm, mean, std)
    
    # Create datasets and loaders
    print("🔧 Creating datasets and data loaders...")
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = create_datasets_and_loaders(
        train_norm, val_norm, test_norm,
        train_mean, val_mean, test_mean,
        train_std, val_std, test_std
    )
    
    # Create model
    print("🧠 Creating model...")
    model = create_model()
    
    # Plot random weights model
    print("🎲 Creating forecast plot with random weights...")
    save_checkpoint_and_plot(model, df, norm, mean, std, 0, is_random=True)
    
    # Train model
    print("🏋️ Starting training...")
    model = train_loop(model, train_loader, val_loader, cfg, df, norm, mean, std)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
