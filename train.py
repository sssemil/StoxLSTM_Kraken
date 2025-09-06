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
import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from config import cfg
from data_utils import (
    seed_all, load_kraken_ohlc, resample_and_fill_1m, compute_rolling_norm,
    create_simple_temporal_split, create_datasets_and_loaders, batch_to_patches
)
from model import create_model, compute_elbo
from plot_utils import plot_ohlcv_forecast, get_actual_data_for_period, calculate_forecast_metrics, print_forecast_metrics, open_plot
from forecast_utils import forecast_from_timestamp, forecast_future

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
    cut = min(len(df) - cfg.T - 1, 1000)
    cut = max(cut, 1)
    forecast_timestamp = df.index[-(cfg.T + cut)]
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


def train_loop(model, train_loader, val_loader, cfg, df, norm, mean, std, patience=5, min_delta=1e-4):
    """Main training loop."""
    # Create TensorBoard writer
    log_dir = f"{cfg.log_dir}/StoxLSTM_{cfg.seed}_{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float('inf')
    best_state = None
    
    # Early stopping parameters
    patience_counter = 0
    
    print(f"Starting training for {cfg.epochs} epochs...")
    print(f"Device: {cfg.device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Beta KL: {cfg.beta_kl}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta}")
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
        
        # Early stopping logic
        if v_mean < best_val - min_delta:
            best_val = v_mean
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            writer.add_scalar('Epoch/Best_Val_Loss', best_val, epoch)
            print(f"  → New best validation loss: {best_val:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter} epochs (best: {best_val:.4f})")
        
        # Save checkpoint and plot forecast after each epoch
        print(f"  📊 Creating forecast plot for epoch {epoch}...")
        save_checkpoint_and_plot(model, df, norm, mean, std, epoch)
        
        # Check for early stopping
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping triggered after {epoch} epochs!")
            print(f"   No improvement in validation loss for {patience} epochs")
            print(f"   Best validation loss: {best_val:.4f}")
            break
        
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
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change for improvement')
    
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
        df, window=cfg.roll_window, min_periods=cfg.min_periods, eps=cfg.epsilon
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
    model = train_loop(model, train_loader, val_loader, cfg, df, norm, mean, std, 
                      patience=args.patience, min_delta=args.min_delta)
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
