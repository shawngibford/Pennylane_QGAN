import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def process_raw_data(raw_data_path, processed_data_path, figures_dir):
    """
    Loads the raw Lucy dataset, performs preprocessing, saves the
    processed data, and generates plots.
    """
    print("Starting data processing...")
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Load and preprocess the data
    lucy = pd.read_csv(raw_data_path, header=None, names=['value'])
    lucy['value'] = pd.to_numeric(lucy['value'], errors='coerce')
    lucy['value'] = lucy['value'].fillna(lucy['value'].rolling(window=10, min_periods=1).mean())
    lucy = lucy.dropna()
    od_tensor = torch.tensor(lucy['value'].values, dtype=torch.float32)
    print('Data shape (total measurements):', od_tensor.shape)

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(od_tensor.numpy())
    plt.title('Lucy Optical Density Time Series')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plot_path = os.path.join(figures_dir, 'lucy_optical_density.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved time series plot to {plot_path}")

    # Display some basic statistics
    print('\nBasic data statistics:')
    print(lucy['value'].describe())

    # Calculate returns
    time_index = np.arange(len(od_tensor))
    lucy_returns = od_tensor[1:] - od_tensor[:-1]
    lucy_log_returns = torch.log(od_tensor[1:]) - torch.log(od_tensor[:-1])
    returns_np = lucy_returns.numpy()
    log_returns_np = lucy_log_returns.numpy()

    # Plot the graphs side-by-side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axes[0].plot(time_index[1:], returns_np)
    axes[0].set_title('Lucy Direct Returns')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Direct Returns')
    axes[0].grid(True)
    axes[1].plot(time_index[1:], log_returns_np)
    axes[1].set_title('Lucy Log Returns')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Log Returns')
    axes[1].grid(True)
    plt.tight_layout()
    returns_plot_path = os.path.join(figures_dir, 'lucy_returns_comparison.png')
    plt.savefig(returns_plot_path)
    plt.close()
    print(f"Saved returns comparison plot to {returns_plot_path}")

    # Print basic statistics of returns
    print("\nDirect Returns Statistics:")
    print(f"Mean: {np.mean(returns_np):.4f}, Std: {np.std(returns_np):.4f}, Skewness: {stats.skew(returns_np):.4f}, Kurtosis: {stats.kurtosis(returns_np):.4f}")

    print("\nLog Returns Statistics:")
    print(f"Mean: {np.mean(log_returns_np):.4f}, Std: {np.std(log_returns_np):.4f}, Skewness: {stats.skew(log_returns_np):.4f}, Kurtosis: {stats.kurtosis(log_returns_np):.4f}")

    # Save log returns for later use
    log_returns_tensor = torch.from_numpy(log_returns_np)
    torch.save(log_returns_tensor, processed_data_path)
    print(f"Processed data (log returns) saved to {processed_data_path}")
    print("Data processing complete.")

def load_processed_data(processed_data_path):
    """
    Loads the preprocessed data for training.
    """
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. "
                              "Run 'python src/data/lucy_dataset.py' to generate it.")
    
    log_returns_tensor = torch.load(processed_data_path)
    return log_returns_tensor

if __name__ == '__main__':
    # This allows running the script to perform preprocessing
    process_raw_data(
        raw_data_path='data/raw/lucy2.csv',
        processed_data_path='data/processed/log_returns.pt',
        figures_dir='reports/figures'
    ) 