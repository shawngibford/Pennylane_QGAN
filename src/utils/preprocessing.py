import numpy as np
import torch
from torch.utils.data import Dataset

def normalize(data):
    """Normalize the data to have zero mean and unit variance."""
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma, mu, sigma

def denormalize(norm_data, mu_original, std_original):
    """Denormalize the data back to original scale."""
    return norm_data * std_original + mu_original

def lambert_w_transform(data, delta=1.0, clip_low=-12.0, clip_high=11.0):
    """Apply Lambert W transformation to make data more Gaussian-like."""
    transformed_data = data.copy()
    transformed_data = np.clip(transformed_data, clip_low, clip_high)
    transformed_data = delta * np.sign(transformed_data) * np.log(1 + np.abs(transformed_data) / delta)
    return transformed_data

def inverse_lambert_w_transform(data, delta=1.0):
    """Inverse Lambert W transformation to recover original scale."""
    return delta * np.sign(data) * (np.exp(np.abs(data) / delta) - 1)

def create_sliding_windows(data, window_size, stride=1):
    """Create sliding windows from the time series data."""
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)

class TimeSeriesDataset(Dataset):
    """Custom PyTorch Dataset for time series data."""
    def __init__(self, data, window_size, stride=1):
        self.data_tensor = torch.tensor(data, dtype=torch.float32)
        self.windows = create_sliding_windows(data, window_size, stride)
        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx] 