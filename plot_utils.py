"""
Shared plotting utilities for StoxLSTM OHLCV visualization.
"""

import matplotlib
matplotlib.use("Agg")  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import subprocess
import os
import platform
import shutil
from typing import Optional, Tuple


def plot_ohlcv_forecast(hist_data: pd.DataFrame, 
                       fut_df: pd.DataFrame, 
                       actual_data: Optional[pd.DataFrame] = None,
                       title: str = "OHLCV Forecast",
                       save_path: Optional[str] = None,
                       show_volume: bool = True,
                       show_plot: bool = False) -> str:
    """
    Create a comprehensive OHLCV forecast plot with historical, forecast, and optional actual data.
    
    Args:
        hist_data: Historical OHLCV data (last 24 hours)
        fut_df: Forecast OHLCV data
        actual_data: Optional actual data for the forecast period (for comparison)
        title: Plot title
        save_path: Optional path to save the plot
        show_volume: Whether to show volume subplot
        show_plot: Whether to display the plot (default: False, saves instead)
        
    Returns:
        Path to the saved plot file
    """
    # Create the plot
    if show_volume:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), 
                                gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.1})
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax2 = None
    
    # Plot OHLC on the main axis
    _plot_ohlc_data(ax1, hist_data, fut_df, actual_data)
    
    # Plot volume if requested
    if show_volume and ax2 is not None:
        _plot_volume_data(ax2, hist_data, fut_df, actual_data)
    
    # Add labels and title
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Deduplicate legend entries
    handles, labels = ax1.get_legend_handles_labels()
    seen = set()
    uniq = [(h,l) for h,l in zip(handles,labels) if (l not in seen) and not seen.add(l)]
    ax1.legend(*zip(*uniq), loc='upper left', fontsize=8)
    
    if ax2 is not None:
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Deduplicate legend entries for volume plot too
        handles, labels = ax2.get_legend_handles_labels()
        seen = set()
        uniq = [(h,l) for h,l in zip(handles,labels) if (l not in seen) and not seen.add(l)]
        ax2.legend(*zip(*uniq), loc='upper left', fontsize=8)
    else:
        ax1.set_xlabel('Time', fontsize=12)
    
    # Add vertical line to separate historical and forecast
    ax1.axvline(x=hist_data.index[-1], color='red', linestyle='-', alpha=0.8, linewidth=2)
    if ax2 is not None:
        ax2.axvline(x=hist_data.index[-1], color='red', linestyle='-', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    
    # Generate default save path if not provided
    if save_path is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"plots/forecast_{timestamp}.png"
    
    # Create plots directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    # Show plot only if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory
    
    return save_path


def _plot_ohlc_data(ax, hist_data: pd.DataFrame, fut_df: pd.DataFrame, actual_data: Optional[pd.DataFrame] = None):
    """Plot OHLC data on the given axis."""
    # Plot historical OHLC
    ax.plot(hist_data.index, hist_data['open'], color='blue', alpha=0.7, linewidth=1, label='Historical Open')
    ax.plot(hist_data.index, hist_data['high'], color='green', alpha=0.7, linewidth=1, label='Historical High')
    ax.plot(hist_data.index, hist_data['low'], color='red', alpha=0.7, linewidth=1, label='Historical Low')
    ax.plot(hist_data.index, hist_data['close'], color='black', alpha=0.9, linewidth=2, label='Historical Close')
    
    # Plot forecast OHLC
    ax.plot(fut_df.index, fut_df['open'], color='blue', alpha=0.7, linewidth=1, linestyle='--', label='Forecast Open')
    ax.plot(fut_df.index, fut_df['high'], color='green', alpha=0.7, linewidth=1, linestyle='--', label='Forecast High')
    ax.plot(fut_df.index, fut_df['low'], color='red', alpha=0.7, linewidth=1, linestyle='--', label='Forecast Low')
    ax.plot(fut_df.index, fut_df['close'], color='black', alpha=0.9, linewidth=2, linestyle='--', label='Forecast Close')
    
    # Plot actual data if available
    if actual_data is not None and len(actual_data) > 0:
        ax.plot(actual_data.index, actual_data['open'], color='blue', alpha=0.9, linewidth=1, label='Actual Open')
        ax.plot(actual_data.index, actual_data['high'], color='green', alpha=0.9, linewidth=1, label='Actual High')
        ax.plot(actual_data.index, actual_data['low'], color='red', alpha=0.9, linewidth=1, label='Actual Low')
        ax.plot(actual_data.index, actual_data['close'], color='black', alpha=1.0, linewidth=1.5, label='Actual Close')


def _plot_volume_data(ax, hist_data: pd.DataFrame, fut_df: pd.DataFrame, actual_data: Optional[pd.DataFrame] = None):
    """Plot volume data on the given axis."""
    # Plot historical volume
    ax.plot(hist_data.index, hist_data['volume'], color='lightblue', alpha=0.8, linewidth=1, label='Historical Volume')
    
    # Plot forecast volume
    ax.plot(fut_df.index, fut_df['volume'], color='orange', alpha=0.8, linewidth=1, linestyle='--', label='Forecast Volume')
    
    # Plot actual volume if available
    if actual_data is not None and len(actual_data) > 0:
        ax.plot(actual_data.index, actual_data['volume'], color='orange', alpha=1.0, linewidth=1, label='Actual Volume')


def get_actual_data_for_period(df: pd.DataFrame, forecast_start, forecast_end) -> Optional[pd.DataFrame]:
    """
    Get actual data for a specific forecast period.
    
    Args:
        df: Full dataset
        forecast_start: Start timestamp of forecast period
        forecast_end: End timestamp of forecast period
        
    Returns:
        Actual data for the forecast period, or None if not available
    """
    try:
        # Inclusive slicing by label is simplest & correct
        actual = df.loc[forecast_start:forecast_end]
        if len(actual):
            print(f"Found actual data for comparison: {len(actual)} samples")
            return actual.copy()
        print("No actual data available for the forecast period")
        return None
    except Exception as e:
        print(f"Could not load actual data for comparison: {e}")
        return None


def calculate_forecast_metrics(forecast_data: pd.DataFrame, actual_data: pd.DataFrame) -> dict:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        forecast_data: Forecast OHLCV data
        actual_data: Actual OHLCV data for the same period
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(forecast_data) != len(actual_data):
        min_len = min(len(forecast_data), len(actual_data))
        forecast_data = forecast_data.iloc[:min_len]
        actual_data = actual_data.iloc[:min_len]
    
    metrics = {}
    
    # Calculate MAE for close prices
    close_mae = np.mean(np.abs(forecast_data['close'] - actual_data['close']))
    metrics['close_mae'] = close_mae
    
    # Calculate percentage error
    close_percentage_error = np.mean(np.abs((forecast_data['close'] - actual_data['close']) / actual_data['close'])) * 100
    metrics['close_percentage_error'] = close_percentage_error
    
    # Calculate final price accuracy
    final_forecast = forecast_data['close'].iloc[-1]
    final_actual = actual_data['close'].iloc[-1]
    final_error = abs(final_forecast - final_actual) / final_actual * 100
    metrics['final_price_error'] = final_error
    
    # Calculate price range comparisons
    forecast_range = forecast_data['close'].max() - forecast_data['close'].min()
    actual_range = actual_data['close'].max() - actual_data['close'].min()
    range_ratio = forecast_range / actual_range if actual_range > 0 else 0
    metrics['range_ratio'] = range_ratio
    
    return metrics


def print_forecast_metrics(metrics: dict) -> None:
    """Print forecast accuracy metrics in a formatted way."""
    print("\n" + "="*50)
    print("FORECAST ACCURACY METRICS")
    print("="*50)
    print(f"Close Price MAE: ${metrics['close_mae']:.2f}")
    print(f"Close Price Error: {metrics['close_percentage_error']:.2f}%")
    print(f"Final Price Error: {metrics['final_price_error']:.2f}%")
    print(f"Price Range Ratio: {metrics['range_ratio']:.2f}")
    print("="*50)


def open_plot(plot_path: str) -> None:
    """Open the saved plot using the system's default image viewer."""
    try:
        if not os.path.exists(plot_path):
            print(f"Plot file not found: {plot_path}")
            return
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", plot_path], check=False)
        elif system == "Windows":
            os.startfile(plot_path)  # type: ignore[attr-defined]
        else:
            opener = shutil.which("xdg-open")
            if opener:
                subprocess.run([opener, plot_path], check=False)
            else:
                print("No system opener found; open the plot manually.")
    except Exception as e:
        print(f"Failed to open plot: {e}")
