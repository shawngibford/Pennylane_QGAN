import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import seaborn as sns

class SinusoidalBenchmarkDataset:
    """
    Generate synthetic time-series data with 5 sinusoidal features for QGAN benchmarking.
    
    Each feature has different characteristics:
    - Feature 0: Low frequency, high amplitude (trend-like)
    - Feature 1: Medium frequency, medium amplitude (seasonal-like)
    - Feature 2: High frequency, low amplitude (noise-like)
    - Feature 3: Mixed frequencies (complex pattern)
    - Feature 4: Correlated with Feature 1 but phase-shifted (interdependent)
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize the dataset generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Define parameters for each feature
        self.feature_params = {
            'feature_0': {
                'frequencies': [0.005, 0.01],  # Very low frequency (trend-like)
                'amplitudes': [2.0, 1.5],
                'phases': [0, np.pi/4],
                'noise_level': 0.1,
                'offset': 0.0
            },
            'feature_1': {
                'frequencies': [0.02, 0.05, 0.08],  # Medium frequency (seasonal-like)
                'amplitudes': [1.5, 0.8, 0.3],
                'phases': [0, np.pi/3, np.pi/2],
                'noise_level': 0.15,
                'offset': 0.5
            },
            'feature_2': {
                'frequencies': [0.1, 0.15, 0.2, 0.25],  # High frequency (detailed patterns)
                'amplitudes': [0.8, 0.4, 0.3, 0.2],
                'phases': [0, np.pi/6, np.pi/4, np.pi/3],
                'noise_level': 0.2,
                'offset': -0.2
            },
            'feature_3': {
                'frequencies': [0.03, 0.12, 0.07],  # Mixed frequencies (complex)
                'amplitudes': [1.0, 0.6, 0.4],
                'phases': [np.pi/4, np.pi/2, 3*np.pi/4],
                'noise_level': 0.18,
                'offset': 0.3
            },
            'feature_4': {
                'frequencies': [0.02, 0.05],  # Correlated with feature_1
                'amplitudes': [1.2, 0.6],
                'phases': [np.pi/2, 3*np.pi/4],  # Phase-shifted
                'noise_level': 0.12,
                'offset': -0.1
            }
        }
    
    def generate_single_feature(self, feature_name: str, length: int, 
                              dt: float = 1.0) -> np.ndarray:
        """
        Generate a single sinusoidal feature with multiple frequency components.
        
        Args:
            feature_name: Name of the feature ('feature_0' to 'feature_4')
            length: Length of the time series
            dt: Time step
            
        Returns:
            Generated time series for the feature
        """
        params = self.feature_params[feature_name]
        t = np.arange(length) * dt
        
        # Initialize with offset
        signal = np.full(length, params['offset'])
        
        # Add sinusoidal components
        for freq, amp, phase in zip(params['frequencies'], 
                                   params['amplitudes'], 
                                   params['phases']):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise
        noise = np.random.normal(0, params['noise_level'], length)
        signal += noise
        
        return signal
    
    def generate_correlated_feature(self, base_feature: np.ndarray, 
                                  correlation: float = 0.7) -> np.ndarray:
        """
        Generate feature_4 as a correlated version of feature_1.
        
        Args:
            base_feature: Base feature to correlate with
            correlation: Correlation strength
            
        Returns:
            Correlated feature
        """
        params = self.feature_params['feature_4']
        length = len(base_feature)
        
        # Create base correlated signal
        correlated_signal = correlation * base_feature
        
        # Add independent component
        independent_component = np.sqrt(1 - correlation**2) * \
                              self.generate_single_feature('feature_4', length)
        
        return correlated_signal + independent_component
    
    def generate_dataset(self, length: int, dt: float = 1.0, 
                        add_correlation: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate the complete 5-feature dataset.
        
        Args:
            length: Length of the time series
            dt: Time step
            add_correlation: Whether to make feature_4 correlated with feature_1
            
        Returns:
            Tuple of (data_array, data_dataframe)
        """
        # Generate independent features
        features = {}
        for i in range(4):  # Features 0-3
            feature_name = f'feature_{i}'
            features[feature_name] = self.generate_single_feature(feature_name, length, dt)
        
        # Generate feature_4 (potentially correlated)
        if add_correlation:
            features['feature_4'] = self.generate_correlated_feature(features['feature_1'])
        else:
            features['feature_4'] = self.generate_single_feature('feature_4', length, dt)
        
        # Create time index
        time_index = np.arange(length) * dt
        
        # Convert to array and DataFrame
        data_array = np.column_stack([features[f'feature_{i}'] for i in range(5)])
        data_df = pd.DataFrame(data_array, 
                              columns=[f'feature_{i}' for i in range(5)],
                              index=time_index)
        
        return data_array, data_df
    
    def add_regime_changes(self, data: np.ndarray, num_changes: int = 2) -> np.ndarray:
        """
        Add regime changes to make the dataset more challenging.
        
        Args:
            data: Input data array
            num_changes: Number of regime changes
            
        Returns:
            Data with regime changes
        """
        length = data.shape[0]
        modified_data = data.copy()
        
        # Add sudden level shifts
        change_points = np.random.choice(range(length//4, 3*length//4), 
                                       size=num_changes, replace=False)
        
        for change_point in change_points:
            # Random shift magnitude for each feature
            shifts = np.random.normal(0, 0.5, data.shape[1])
            modified_data[change_point:] += shifts
        
        return modified_data
    
    def normalize_data(self, data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
        """
        Normalize the dataset.
        
        Args:
            data: Input data array
            method: Normalization method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'standard':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            normalized_data = (data - mean) / std
            params = {'mean': mean, 'std': std, 'method': 'standard'}
            
        elif method == 'minmax':
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            normalized_data = (data - min_vals) / (max_vals - min_vals)
            params = {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
            
        elif method == 'robust':
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            normalized_data = (data - median) / mad
            params = {'median': median, 'mad': mad, 'method': 'robust'}
            
        else:
            raise ValueError("Method must be 'standard', 'minmax', or 'robust'")
        
        return normalized_data, params
    
    def create_windows(self, data: np.ndarray, window_size: int, 
                      step_size: int = 1) -> np.ndarray:
        """
        Create sliding windows from the time series data.
        
        Args:
            data: Input data array
            window_size: Size of each window
            step_size: Step size between windows
            
        Returns:
            Array of windows with shape (num_windows, window_size, num_features)
        """
        length, num_features = data.shape
        num_windows = (length - window_size) // step_size + 1
        
        windows = np.zeros((num_windows, window_size, num_features))
        
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            windows[i] = data[start_idx:end_idx]
        
        return windows
    
    def plot_dataset(self, data_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot the generated dataset.
        
        Args:
            data_df: DataFrame containing the dataset
            figsize: Figure size
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Sinusoidal Benchmark Dataset', fontsize=16)
        
        # Individual feature plots
        for i in range(5):
            row = i // 2
            col = i % 2
            axes[row, col].plot(data_df.index, data_df[f'feature_{i}'], 
                               linewidth=1, alpha=0.8)
            axes[row, col].set_title(f'Feature {i}')
            axes[row, col].set_xlabel('Time')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)
        
        # Correlation heatmap
        correlation_matrix = data_df.corr()
        axes[2, 1].clear()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   center=0, ax=axes[2, 1])
        axes[2, 1].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_info(self, data: np.ndarray) -> dict:
        """
        Get information about the generated dataset.
        
        Args:
            data: Dataset array
            
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'shape': data.shape,
            'length': data.shape[0],
            'num_features': data.shape[1],
            'mean_per_feature': np.mean(data, axis=0),
            'std_per_feature': np.std(data, axis=0),
            'min_per_feature': np.min(data, axis=0),
            'max_per_feature': np.max(data, axis=0),
            'correlation_matrix': np.corrcoef(data.T)
        }


def create_benchmark_dataset(length: int = 5000, window_size: int = 50, 
                           normalize: bool = True, add_regime_changes: bool = False,
                           save_path: Optional[str] = None, save_csv: Optional[str] = None) -> dict:
    """
    Convenience function to create a complete benchmark dataset for QGAN testing.
    
    Args:
        length: Length of the time series
        window_size: Size of windows for QGAN training
        normalize: Whether to normalize the data
        add_regime_changes: Whether to add regime changes
        save_path: Path to save the dataset (optional)
        save_csv: Path to save as CSV file (optional)
        
    Returns:
        Dictionary containing the dataset and metadata
    """
    # Create dataset generator
    generator = SinusoidalBenchmarkDataset(seed=42)
    
    # Generate raw data
    data_array, data_df = generator.generate_dataset(length)
    
    # Add regime changes if requested
    if add_regime_changes:
        data_array = generator.add_regime_changes(data_array)
        data_df = pd.DataFrame(data_array, 
                              columns=[f'feature_{i}' for i in range(5)],
                              index=data_df.index)
    
    # Normalize if requested
    norm_params = None
    if normalize:
        data_array, norm_params = generator.normalize_data(data_array)
        data_df = pd.DataFrame(data_array, 
                              columns=[f'feature_{i}' for i in range(5)],
                              index=data_df.index)
    
    # Create windows
    windows = generator.create_windows(data_array, window_size)
    
    # Get dataset info
    info = generator.get_dataset_info(data_array)
    
    # Create result dictionary
    result = {
        'raw_data': data_array,
        'data_df': data_df,
        'windows': windows,
        'window_size': window_size,
        'normalization_params': norm_params,
        'dataset_info': info,
        'generator': generator
    }
    
    # Save if requested
    if save_path:
        np.savez(save_path, 
                raw_data=data_array,
                windows=windows,
                normalization_params=norm_params if norm_params else {},
                dataset_info=info)
        print(f"Dataset saved to {save_path}")
    
    # Save as CSV if requested
    if save_csv:
        data_df.to_csv(save_csv, index=True)
        print(f"Dataset saved as CSV to {save_csv}")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("Creating benchmark dataset...")
    
    # Create dataset
    dataset = create_benchmark_dataset(
        length=2000,
        window_size=32,
        normalize=True,
        add_regime_changes=False,
        save_path="benchmark_dataset.npz"
    )
    
    # Display info
    print(f"Dataset shape: {dataset['raw_data'].shape}")
    print(f"Number of windows: {dataset['windows'].shape[0]}")
    print(f"Window shape: {dataset['windows'].shape[1:]}")
    
    # Plot the dataset
    dataset['generator'].plot_dataset(dataset['data_df'])
    
    print("\nDataset statistics:")
    for key, value in dataset['dataset_info'].items():
        if key != 'correlation_matrix':
            print(f"{key}: {value}") 