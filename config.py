"""
Configuration management for StoxLSTM model.
"""
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class Config:
    # Data paths
    csv_path: str = "data/BTCUSD_1.csv"
    time_col: str = "timestamp"
    cols: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
    resample_1m: bool = True
    tz: str = "UTC"

    # Windows
    L: int = 1440  # lookback (24h)
    T: int = 2880  # horizon (48h) - up to 48 hours into the future
    stride_windows: int = 60  # shift between training windows (1h)
    
    # Patching
    P: int = 56
    S: int = 24

    # Normalization
    roll_days: int = 30
    min_periods: int = 43200  # 30d * 24h * 60m
    epsilon: float = 1e-8

    # Model
    d_model: int = 64
    d_latent: int = 16
    num_layers: int = 1
    dropout: float = 0.05

    # Training
    batch_size: int = 64
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.0
    beta_kl: float = 5.0  # KL weight (tune 1–50)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 1337
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.T > 4320:  # 72 hours - relaxed from 48h to match README
            raise ValueError("T (forecast horizon) should not exceed 4320 minutes (72 hours)")
        
        if self.L < 1440:  # 24 hours minimum
            print(f"Warning: L (lookback) is {self.L} minutes, consider using at least 1440 (24h)")

# Global config instance
cfg = Config()
