"""
Dataset Generator for Quantum Synthetic Data Project
===================================================

This module generates the 9 required datasets as specified in the project requirements:
- 3 datasets with Gaussian log-distributions
- 3 datasets with non-log distributions  
- 3 datasets with multi-modal log distributions
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import lognorm, pareto, expon, norm
import os
import json

class DatasetGenerator:
    """
    Generator for the 9 required datasets with different distribution characteristics.
    """
    
    def __init__(self, sample_size: int = 10000, n_features: int = 5, 
                 save_path: str = "data/generated"):
        """
        Initialize the dataset generator.
        
        Args:
            sample_size: Number of samples per dataset
            n_features: Number of features per sample
            save_path: Path to save generated datasets
        """
        self.sample_size = sample_size
        self.n_features = n_features
        self.save_path = save_path
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Dataset specifications
        self.dataset_specs = {
            # Gaussian Log-Distribution Datasets
            'A': {'type': 'gaussian_log', 'description': 'Pure Gaussian with log transformation'},
            'B': {'type': 'gaussian_log', 'description': 'Gaussian with slight skewness'},
            'C': {'type': 'gaussian_log', 'description': 'Gaussian with controlled variance'},
            
            # Non-Log Distribution Datasets
            'D': {'type': 'non_log', 'description': 'Heavy-tailed distribution (Pareto)'},
            'E': {'type': 'non_log', 'description': 'Multi-modal distribution'},
            'F': {'type': 'non_log', 'description': 'Exponential distribution'},
            
            # Multi-Modal Log Distribution Datasets
            'G': {'type': 'multimodal_log', 'description': 'Bimodal log-normal'},
            'H': {'type': 'multimodal_log', 'description': 'Trimodal with varying modes'},
            'I': {'type': 'multimodal_log', 'description': 'Complex multi-modal with noise'}
        }
    
    def generate_gaussian_log_datasets(self) -> Dict[str, np.ndarray]:
        """
        Generate 3 datasets with Gaussian log-distributions.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of datasets A, B, C
        """
        datasets = {}
        
        # Dataset A: Pure Gaussian log
        datasets['A'] = np.random.lognormal(
            mean=0, sigma=1, size=(self.sample_size, self.n_features)
        )
        
        # Dataset B: Slightly skewed Gaussian log
        datasets['B'] = np.random.lognormal(
            mean=0.5, sigma=1.2, size=(self.sample_size, self.n_features)
        )
        
        # Dataset C: Controlled variance Gaussian log
        datasets['C'] = np.random.lognormal(
            mean=0, sigma=0.8, size=(self.sample_size, self.n_features)
        )
        
        return datasets
    
    def generate_non_log_datasets(self) -> Dict[str, np.ndarray]:
        """
        Generate 3 datasets with non-log distributions.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of datasets D, E, F
        """
        datasets = {}
        
        # Dataset D: Pareto distribution (heavy-tailed)
        datasets['D'] = np.random.pareto(
            a=2.0, size=(self.sample_size, self.n_features)
        )
        
        # Dataset E: Multi-modal distribution
        # Mix of two normal distributions
        n1 = self.sample_size // 2
        n2 = self.sample_size - n1
        
        mode1 = np.random.normal(0, 1, (n1, self.n_features))
        mode2 = np.random.normal(5, 1, (n2, self.n_features))
        datasets['E'] = np.vstack([mode1, mode2])
        
        # Shuffle the data
        np.random.shuffle(datasets['E'])
        
        # Dataset F: Exponential distribution
        datasets['F'] = np.random.exponential(
            scale=2.0, size=(self.sample_size, self.n_features)
        )
        
        return datasets
    
    def generate_multimodal_log_datasets(self) -> Dict[str, np.ndarray]:
        """
        Generate 3 datasets with multi-modal log distributions.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of datasets G, H, I
        """
        datasets = {}
        
        # Dataset G: Bimodal log-normal
        n1 = self.sample_size // 2
        n2 = self.sample_size - n1
        
        mode1 = np.random.lognormal(mean=0, sigma=1, size=(n1, self.n_features))
        mode2 = np.random.lognormal(mean=2, sigma=1, size=(n2, self.n_features))
        datasets['G'] = np.vstack([mode1, mode2])
        np.random.shuffle(datasets['G'])
        
        # Dataset H: Trimodal
        n1 = self.sample_size // 3
        n2 = self.sample_size // 3
        n3 = self.sample_size - n1 - n2
        
        mode1 = np.random.lognormal(mean=0, sigma=0.8, size=(n1, self.n_features))
        mode2 = np.random.lognormal(mean=1.5, sigma=1, size=(n2, self.n_features))
        mode3 = np.random.lognormal(mean=3, sigma=1.2, size=(n3, self.n_features))
        datasets['H'] = np.vstack([mode1, mode2, mode3])
        np.random.shuffle(datasets['H'])
        
        # Dataset I: Complex multi-modal with noise
        n1 = int(0.4 * self.sample_size)
        n2 = int(0.3 * self.sample_size)
        n3 = self.sample_size - n1 - n2
        
        mode1 = np.random.lognormal(mean=0, sigma=1, size=(n1, self.n_features))
        mode2 = np.random.lognormal(mean=2, sigma=0.8, size=(n2, self.n_features))
        mode3 = np.random.lognormal(mean=4, sigma=1.5, size=(n3, self.n_features))
        
        # Combine modes and add noise
        combined = np.vstack([mode1, mode2, mode3])
        noise = np.random.normal(0, 0.1, (self.sample_size, self.n_features))
        datasets['I'] = combined + noise
        np.random.shuffle(datasets['I'])
        
        return datasets
    
    def generate_all_datasets(self) -> Dict[str, np.ndarray]:
        """
        Generate all 9 datasets.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of all datasets A through I
        """
        print("Generating Gaussian log-distribution datasets (A, B, C)...")
        gaussian_log_datasets = self.generate_gaussian_log_datasets()
        
        print("Generating non-log distribution datasets (D, E, F)...")
        non_log_datasets = self.generate_non_log_datasets()
        
        print("Generating multi-modal log distribution datasets (G, H, I)...")
        multimodal_log_datasets = self.generate_multimodal_log_datasets()
        
        # Combine all datasets
        all_datasets = {}
        all_datasets.update(gaussian_log_datasets)
        all_datasets.update(non_log_datasets)
        all_datasets.update(multimodal_log_datasets)
        
        print(f"Generated {len(all_datasets)} datasets with {self.sample_size} samples each")
        
        return all_datasets
    
    def validate_datasets(self, datasets: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Validate that all datasets meet specifications.
        
        Args:
            datasets: Dictionary of datasets to validate
        
        Returns:
            Dict[str, Dict]: Validation results for each dataset
        """
        validation_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"Validating dataset {dataset_name}...")
            
            validation_results[dataset_name] = {
                'sample_size': len(data),
                'dimensionality': data.shape[1],
                'distribution_type': self._analyze_distribution(data),
                'log_normality': self._test_log_normality(data),
                'multimodality': self._test_multimodality(data),
                'basic_stats': self._calculate_basic_stats(data),
                'specification': self.dataset_specs[dataset_name]
            }
        
        return validation_results
    
    def _analyze_distribution(self, data: np.ndarray) -> Dict:
        """
        Analyze the distribution of the data.
        
        Args:
            data: Dataset to analyze
        
        Returns:
            Dict: Distribution analysis results
        """
        # Test for normality
        normality_tests = []
        for i in range(data.shape[1]):
            stat, p_value = stats.normaltest(data[:, i])
            normality_tests.append({'statistic': stat, 'p_value': p_value})
        
        # Test for log-normality
        log_normality_tests = []
        for i in range(data.shape[1]):
            log_data = np.log(data[:, i] + 1e-8)  # Add small constant to avoid log(0)
            stat, p_value = stats.normaltest(log_data)
            log_normality_tests.append({'statistic': stat, 'p_value': p_value})
        
        return {
            'normality_tests': normality_tests,
            'log_normality_tests': log_normality_tests,
            'skewness': [stats.skew(data[:, i]) for i in range(data.shape[1])],
            'kurtosis': [stats.kurtosis(data[:, i]) for i in range(data.shape[1])]
        }
    
    def _test_log_normality(self, data: np.ndarray) -> Dict:
        """
        Test for log-normality of the data.
        
        Args:
            data: Dataset to test
        
        Returns:
            Dict: Log-normality test results
        """
        results = {}
        
        for i in range(data.shape[1]):
            log_data = np.log(data[:, i] + 1e-8)
            
            # Kolmogorov-Smirnov test against log-normal
            ks_stat, ks_p = stats.kstest(log_data, 'norm')
            
            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = stats.anderson(log_data)
            
            results[f'feature_{i}'] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'anderson_statistic': ad_stat,
                'anderson_critical': ad_critical,
                'anderson_significance': ad_significance
            }
        
        return results
    
    def _test_multimodality(self, data: np.ndarray) -> Dict:
        """
        Test for multimodality in the data.
        
        Args:
            data: Dataset to test
        
        Returns:
            Dict: Multimodality test results
        """
        results = {}
        
        for i in range(data.shape[1]):
            # Dip test for multimodality
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data[:, i])
                x_range = np.linspace(data[:, i].min(), data[:, i].max(), 1000)
                density = kde(x_range)
                
                # Simple peak detection
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(density, height=np.max(density) * 0.1)
                n_peaks = len(peaks)
                
            except ImportError:
                n_peaks = 1  # Fallback
            
            results[f'feature_{i}'] = {
                'n_peaks': n_peaks,
                'is_multimodal': n_peaks > 1
            }
        
        return results
    
    def _calculate_basic_stats(self, data: np.ndarray) -> Dict:
        """
        Calculate basic statistics for the data.
        
        Args:
            data: Dataset to analyze
        
        Returns:
            Dict: Basic statistics
        """
        return {
            'mean': np.mean(data, axis=0).tolist(),
            'std': np.std(data, axis=0).tolist(),
            'min': np.min(data, axis=0).tolist(),
            'max': np.max(data, axis=0).tolist(),
            'median': np.median(data, axis=0).tolist(),
            'q25': np.percentile(data, 25, axis=0).tolist(),
            'q75': np.percentile(data, 75, axis=0).tolist()
        }
    
    def save_datasets(self, datasets: Dict[str, np.ndarray], 
                     validation_results: Dict[str, Dict] = None) -> None:
        """
        Save datasets and validation results.
        
        Args:
            datasets: Dictionary of datasets to save
            validation_results: Validation results to save
        """
        # Save datasets as numpy arrays
        for name, data in datasets.items():
            np.save(os.path.join(self.save_path, f"dataset_{name}.npy"), data)
            
            # Also save as CSV for easy inspection
            df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
            df.to_csv(os.path.join(self.save_path, f"dataset_{name}.csv"), index=False)
        
        # Save validation results
        if validation_results:
            with open(os.path.join(self.save_path, "validation_results.json"), 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
        
        # Save dataset metadata
        metadata = {
            'sample_size': self.sample_size,
            'n_features': self.n_features,
            'dataset_specs': self.dataset_specs,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(self.save_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Datasets saved to {self.save_path}")
    
    def plot_datasets(self, datasets: Dict[str, np.ndarray], 
                     save_plots: bool = True) -> None:
        """
        Create visualization plots for all datasets.
        
        Args:
            datasets: Dictionary of datasets to plot
            save_plots: Whether to save plots to file
        """
        n_datasets = len(datasets)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(datasets.items()):
            ax = axes[idx]
            
            # Plot histograms for each feature
            for i in range(min(3, data.shape[1])):  # Plot first 3 features
                ax.hist(data[:, i], bins=50, alpha=0.7, label=f'Feature {i}')
            
            ax.set_title(f'Dataset {name}: {self.dataset_specs[name]["description"]}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(os.path.join(self.save_path, "dataset_distributions.png"), 
                       dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to {self.save_path}")
        
        plt.show()
    
    def load_datasets(self) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Load previously generated datasets.
        
        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Dict]]: Datasets and validation results
        """
        datasets = {}
        validation_results = {}
        
        # Load datasets
        for name in self.dataset_specs.keys():
            data_file = os.path.join(self.save_path, f"dataset_{name}.npy")
            if os.path.exists(data_file):
                datasets[name] = np.load(data_file)
        
        # Load validation results
        validation_file = os.path.join(self.save_path, "validation_results.json")
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                validation_results = json.load(f)
        
        return datasets, validation_results

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = DatasetGenerator(sample_size=10000, n_features=5)
    
    # Generate all datasets
    print("Generating all 9 datasets...")
    datasets = generator.generate_all_datasets()
    
    # Validate datasets
    print("Validating datasets...")
    validation_results = generator.validate_datasets(datasets)
    
    # Save datasets and results
    print("Saving datasets and results...")
    generator.save_datasets(datasets, validation_results)
    
    # Create plots
    print("Creating visualization plots...")
    generator.plot_datasets(datasets)
    
    # Print summary
    print("\nDataset Generation Summary:")
    print("=" * 50)
    for name, spec in generator.dataset_specs.items():
        print(f"Dataset {name}: {spec['description']}")
        if name in validation_results:
            stats = validation_results[name]['basic_stats']
            print(f"  - Sample size: {validation_results[name]['sample_size']}")
            print(f"  - Features: {validation_results[name]['dimensionality']}")
            print(f"  - Mean range: [{min(stats['mean']):.3f}, {max(stats['mean']):.3f}]")
            print(f"  - Std range: [{min(stats['std']):.3f}, {max(stats['std']):.3f}]")
        print() 