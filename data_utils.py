"""
Data utilities for StoxLSTM model.
Handles data loading, preprocessing, normalization, and dataset creation.
"""
import math
import random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import cfg

def seed_all(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_kraken_ohlc(csv_path, time_col="timestamp", cols=None, tz="UTC"):
    """
    Load Kraken OHLCV(T) CSV data into a pandas DataFrame with a proper DatetimeIndex.

    Args:
        csv_path: Path to CSV file
        time_col: Column containing timestamps (default "timestamp")
        cols: List of OHLCV columns to keep (default: all except time)
        tz: Target timezone (default "UTC")

    Returns:
        DataFrame indexed by tz-aware datetime
    """
    df = pd.read_csv(csv_path)

    # Autodetect if user passed "time" but file has "timestamp"
    if time_col not in df.columns:
        if "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            raise KeyError(f"{time_col} not found in CSV columns: {df.columns.tolist()}")

    s = df[time_col]
    if np.issubdtype(s.dtype, np.number):
        # heuristic: >= 10^12 → ms; else seconds
        unit = "ms" if float(s.iloc[0]) >= 1_000_000_000_000 else "s"
        dt = pd.to_datetime(s, unit=unit, utc=True).dt.tz_convert(tz)
    else:
        dt = pd.to_datetime(s, utc=True, errors="raise").dt.tz_convert(tz)

    df = df.copy()
    df.index = dt
    df.index.name = "time"

    if cols is not None:
        keep = [c for c in cols if c in df.columns]
        df = df[keep]

    return df

def resample_and_fill_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 1m, forward-fill prices; set missing volume to 0."""
    # Resample to 1m, forward-fill prices; set missing volume to 0
    # OHLC rules for upsampling:
    o = df['open'].resample('1min').first()
    h = df['high'].resample('1min').max()
    l = df['low'].resample('1min').min()
    c = df['close'].resample('1min').last()
    v = df['volume'].resample('1min').sum()

    # Fill gaps: if no trades, carry forward last price; volume=0
    o = o.ffill()
    h = pd.concat([h, c], axis=1).max(axis=1).ffill()
    l = pd.concat([l, c], axis=1).min(axis=1).ffill()
    c = c.ffill()
    v = v.fillna(0.0)

    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})
    return out

def compute_rolling_norm(df: pd.DataFrame, window=2880, min_periods=2880, eps=1e-8):
    """
    Compute rolling normalization using 48h window with 50% expanded bounds.
    Each t uses min/max from t-48h..t, then expands bounds by 50% on each side.
    """
    # Use 48h rolling window to get min/max
    roll_min = df.rolling(window=window, min_periods=min_periods).min()
    roll_max = df.rolling(window=window, min_periods=min_periods).max()
    
    # Fallback for early indices: use expanding stats
    exp_min = df.expanding(min_periods=2).min()
    exp_max = df.expanding(min_periods=2).max()
    
    min_vals = roll_min.where(roll_min.notna(), exp_min)
    max_vals = roll_max.where(roll_max.notna(), exp_max)
    
    # Expand bounds by 50% on each side
    range_vals = max_vals - min_vals
    expanded_min = min_vals - 0.5 * range_vals
    expanded_max = max_vals + 0.5 * range_vals
    
    # Compute mean and std from expanded bounds
    mean = (expanded_min + expanded_max) / 2
    std = (expanded_max - expanded_min) / 6  # Approximate std from range (assuming normal-ish distribution)
    
    # Clip std to avoid very large normalized values in flat regions
    std = std.clip(lower=1e-6)

    norm = (df - mean) / (std + eps)
    return norm, mean, std

def make_windows_index(n: int, L: int, T: int, stride: int) -> List[Tuple[int, int, int]]:
    """Generate window indices for time series data."""
    # windows: [t-L, t) -> predict [t, t+T)
    # last usable t is n-T
    out = []
    t = L
    while t + T <= n:
        out.append((t - L, t, t + T))
        t += stride
    return out

def to_patches(x: torch.Tensor, P: int, S: int) -> torch.Tensor:
    """
    Convert time series to patches.
    x: [B, L, 1] (single channel)
    returns patches: [B, Np, P, 1], where Np = ceil((L - P)/S) + 1
    """
    B, L, C = x.shape
    assert C == 1
    # Zero pad to fit
    pad_right = (math.ceil((L - P) / S) * S + P) - L if L >= P else P - L
    if pad_right < 0: pad_right = 0
    xpad = F.pad(x.transpose(1, 2), (0, pad_right))  # [B,1,L+pad]
    xpad = xpad.unfold(dimension=2, size=P, step=S).transpose(1, 2)  # [B, Np, 1, P]
    xpad = xpad.transpose(2, 3)  # [B, Np, P, 1]
    return xpad

class OHLCVDataset(Dataset):
    """Dataset for OHLCV time series data."""
    
    def __init__(self, df_norm: pd.DataFrame, df_mean: pd.DataFrame, df_std: pd.DataFrame,
                 L: int, T: int, stride: int, cols: Tuple[str, ...]):
        self.df_norm = df_norm
        self.df_mean = df_mean
        self.df_std = df_std
        self.L = L
        self.T = T
        self.cols = cols
        n = len(df_norm)
        self.indexes = make_windows_index(n, L, T, stride)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        s, t, u = self.indexes[idx]
        # x_hist: [L, C], y_future: [T, C]
        x = self.df_norm.iloc[s:t][list(self.cols)].values.astype(np.float32)  # [L, C]
        y = self.df_norm.iloc[t:u][list(self.cols)].values.astype(np.float32)  # [T, C]
        mu_hist = self.df_mean.iloc[s:t][list(self.cols)].values.astype(np.float32)
        sd_hist = self.df_std.iloc[s:t][list(self.cols)].values.astype(np.float32)
        mu_fut = self.df_mean.iloc[t:u][list(self.cols)].values.astype(np.float32)
        sd_fut = self.df_std.iloc[t:u][list(self.cols)].values.astype(np.float32)
        return {
            "x": x, "y": y,
            "mu_hist": mu_hist, "sd_hist": sd_hist,
            "mu_fut": mu_fut, "sd_fut": sd_fut,
        }

def batch_to_patches(batch, P, S):
    """
    Convert batch to patches for each channel.
    Accepts either dict of batched arrays/tensors or list of dicts.
    Returns: patches_hist_list, patches_all_list, target_patch_vals (lists of tensors)
    """
    # Get x,y as torch tensors shaped [B, L, C] and [B, T, C]
    if isinstance(batch, dict):
        # Default collate path
        x = torch.as_tensor(batch["x"])  # [B, L, C]
        y = torch.as_tensor(batch["y"])  # [B, T, C]
    else:
        # Fallback: batch is a list of dicts
        x = torch.as_tensor(np.stack([b["x"] for b in batch], axis=0))
        y = torch.as_tensor(np.stack([b["y"] for b in batch], axis=0))

    B, L, C = x.shape
    patches_hist_list = []
    patches_all_list = []
    target_patch_vals = []

    for c in range(C):
        x_c = x[:, :, c].unsqueeze(-1)  # [B, L, 1]
        ph = to_patches(x_c, P, S)  # [B, Np_hist, P, 1]
        x_all = torch.cat([x_c, y[:, :, c].unsqueeze(-1)], dim=1)  # [B, L+T, 1]
        pa = to_patches(x_all, P, S)  # [B, Np_all, P, 1]
        
        # Fix: Use future patch values as targets for proper forecasting
        # First Nh patches are history, next are future
        Nh = ph.size(1)
        tgt = pa[:, 1:Nh+1, :, :].mean(dim=2).squeeze(-1)  # [B, Np_hist] - predict next patch mean

        patches_hist_list.append(ph)
        patches_all_list.append(pa)
        target_patch_vals.append(tgt)

    return patches_hist_list, patches_all_list, target_patch_vals

def create_simple_temporal_split(norm, mean, std, train_ratio=0.8, val_ratio=0.1):
    """
    Create simple temporal train/val/test split.
    
    Args:
        norm, mean, std: Normalized dataframes
        train_ratio: Fraction for training (default 0.8)
        val_ratio: Fraction for validation (default 0.1)
    
    Returns:
        train_norm, val_norm, test_norm, train_mean, val_mean, test_mean, train_std, val_std, test_std
    """
    n = len(norm)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    print(f"Creating temporal split:")
    print(f"  Total: {n} samples")
    print(f"  Train: 0 to {train_end} ({train_end} samples)")
    print(f"  Val: {train_end} to {val_end} ({val_end - train_end} samples)")
    print(f"  Test: {val_end} to {n} ({n - val_end} samples)")
    
    # Split data
    train_norm = norm.iloc[:train_end]
    val_norm = norm.iloc[train_end:val_end]
    test_norm = norm.iloc[val_end:]
    
    train_mean = mean.iloc[:train_end]
    val_mean = mean.iloc[train_end:val_end]
    test_mean = mean.iloc[val_end:]
    
    train_std = std.iloc[:train_end]
    val_std = std.iloc[train_end:val_end]
    test_std = std.iloc[val_end:]
    
    return (train_norm, val_norm, test_norm, 
            train_mean, val_mean, test_mean, 
            train_std, val_std, test_std)

def create_datasets_and_loaders(train_norm, val_norm, test_norm, 
                               train_mean, val_mean, test_mean,
                               train_std, val_std, test_std):
    """Create datasets and data loaders."""
    
    # Create datasets
    train_ds = OHLCVDataset(train_norm, train_mean, train_std, cfg.L, cfg.T, cfg.stride_windows, cfg.cols)
    val_ds = OHLCVDataset(val_norm, val_mean, val_std, cfg.L, cfg.T, cfg.stride_windows, cfg.cols)
    test_ds = OHLCVDataset(test_norm, test_mean, test_std, cfg.L, cfg.T, cfg.stride_windows, cfg.cols)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    
    # Calculate batch counts
    train_batches = math.ceil(len(train_ds) / cfg.batch_size)
    val_batches = math.ceil(len(val_ds) / cfg.batch_size)
    test_batches = math.ceil(len(test_ds) / cfg.batch_size)
    
    print(f"\nBatch counts per epoch:")
    print(f"  Train: {train_batches} batches")
    print(f"  Val: {val_batches} batches")
    print(f"  Test: {test_batches} batches")
    
    if len(val_ds) == 0:
        print("⚠️  WARNING: Validation dataset is empty!")
    elif train_batches < 10:
        print("⚠️  WARNING: Very few training batches per epoch.")
    else:
        print("✅ Dataset sizes look good!")
    
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
