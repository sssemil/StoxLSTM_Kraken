"""
Shared forecasting utilities for StoxLSTM model.
Consolidates duplicate forecast implementations from train.py and predict.py.
"""
import math
import numpy as np
import pandas as pd
import torch
from data_utils import to_patches
from config import cfg


@torch.no_grad()
def forecast_from_timestamp(model, df_norm, df_mean, df_std, cols, timestamp, L, T, P, S, device):
    """
    Create forecast from a specific timestamp.
    
    Args:
        model: Trained StoxLSTM model
        df_norm: Normalized dataframe
        df_mean: Mean values for denormalization
        df_std: Standard deviation values for denormalization
        cols: Column names
        timestamp: Timestamp to forecast from
        L: Lookback window size
        T: Forecast horizon
        P: Patch size
        S: Patch stride
        device: Device to run on
        
    Returns:
        fut_df: Forecast dataframe
        hist_start: Start index of history window
        hist_end: End index of history window
    """
    # Find the timestamp index
    if isinstance(timestamp, str):
        ts = pd.to_datetime(timestamp)
    else:
        ts = timestamp
    
    # Make timezone aware if needed
    if ts.tz is None:
        ts = ts.tz_localize(df_norm.index.tz or 'UTC')
    else:
        ts = ts.tz_convert(df_norm.index.tz or 'UTC')
    
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


@torch.no_grad()
def forecast_future(model, df_norm, df_mean, df_std, cols, L, T, P, S, device):
    """
    Create forecast for the next T minutes from the end of the dataset.
    
    Args:
        model: Trained StoxLSTM model
        df_norm: Normalized dataframe
        df_mean: Mean values for denormalization
        df_std: Standard deviation values for denormalization
        cols: Column names
        L: Lookback window size
        T: Forecast horizon
        P: Patch size
        S: Patch stride
        device: Device to run on
        
    Returns:
        fut_df: Forecast dataframe
    """
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
