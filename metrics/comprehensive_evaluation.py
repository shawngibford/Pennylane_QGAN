"""
Comprehensive evaluation module for QGAN performance assessment.

This module provides high-level functions to evaluate generated time series
against real data using all available metrics, with reporting and visualization
capabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings
from datetime import datetime

from .basic_metrics import calculate_basic_metrics
from .statistical_metrics import calculate_statistical_metrics
from .time_series_metrics import calculate_time_series_metrics


class MetricsCalculator:
    """
    Comprehensive metrics calculator for QGAN evaluation.
    """
    
    def __init__(self, bins: int = 50, max_lags: int = 20, 
                 seasonal_period: Optional[int] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            bins: Number of bins for histogram-based metrics
            max_lags: Maximum lags for autocorrelation analysis
            seasonal_period: Expected seasonal period
        """
        self.bins = bins
        self.max_lags = max_lags
        self.seasonal_period = seasonal_period
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.
        
        Args:
            y_true: True/real data
            y_pred: Generated/predicted data
            y_train: Training data (for MASE calculation)
            
        Returns:
            Dictionary with all metric values
        """
        all_metrics = {}
        
        # Basic regression metrics
        try:
            basic_metrics = calculate_basic_metrics(y_true, y_pred, y_train)
            all_metrics.update(basic_metrics)
        except Exception as e:
            warnings.warn(f"Error calculating basic metrics: {e}")
        
        # Statistical metrics
        try:
            statistical_metrics = calculate_statistical_metrics(y_true, y_pred, self.bins)
            all_metrics.update(statistical_metrics)
        except Exception as e:
            warnings.warn(f"Error calculating statistical metrics: {e}")
        
        # Time series metrics
        try:
            ts_metrics = calculate_time_series_metrics(y_true, y_pred, 
                                                     self.max_lags, self.seasonal_period)
            all_metrics.update(ts_metrics)
        except Exception as e:
            warnings.warn(f"Error calculating time series metrics: {e}")
        
        return all_metrics
    
    def evaluate_multivariate(self, y_true: np.ndarray, y_pred: np.ndarray,
                            feature_names: Optional[List[str]] = None,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multivariate time series (feature by feature).
        
        Args:
            y_true: True data (n_samples, n_features)
            y_pred: Generated data (n_samples, n_features)
            feature_names: Names of features
            y_train: Training data
            
        Returns:
            Dictionary with metrics for each feature
        """
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        
        n_features = y_true.shape[1]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        results = {}
        
        for i, feature_name in enumerate(feature_names):
            y_true_feature = y_true[:, i]
            y_pred_feature = y_pred[:, i]
            y_train_feature = y_train[:, i] if y_train is not None else None
            
            feature_metrics = self.calculate_all_metrics(
                y_true_feature, y_pred_feature, y_train_feature
            )
            results[feature_name] = feature_metrics
        
        return results
    
    def calculate_aggregate_metrics(self, feature_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all features.
        
        Args:
            feature_metrics: Metrics for each feature
            
        Returns:
            Aggregated metrics
        """
        if not feature_metrics:
            return {}
        
        # Get all metric names
        all_metric_names = set()
        for feature_metrics_dict in feature_metrics.values():
            all_metric_names.update(feature_metrics_dict.keys())
        
        aggregate_metrics = {}
        
        for metric_name in all_metric_names:
            values = []
            for feature_name, metrics in feature_metrics.items():
                if metric_name in metrics and not np.isnan(metrics[metric_name]) and not np.isinf(metrics[metric_name]):
                    values.append(metrics[metric_name])
            
            if values:
                aggregate_metrics[f'{metric_name}_mean'] = np.mean(values)
                aggregate_metrics[f'{metric_name}_std'] = np.std(values)
                aggregate_metrics[f'{metric_name}_min'] = np.min(values)
                aggregate_metrics[f'{metric_name}_max'] = np.max(values)
        
        return aggregate_metrics


def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        y_train: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None,
                        bins: int = 50, max_lags: int = 20,
                        seasonal_period: Optional[int] = None) -> Dict:
    """
    Comprehensive evaluation function that calculates all metrics.
    
    Args:
        y_true: True/real data
        y_pred: Generated/predicted data  
        y_train: Training data (optional)
        feature_names: Names of features (for multivariate data)
        bins: Number of bins for histogram-based metrics
        max_lags: Maximum lags for autocorrelation
        seasonal_period: Expected seasonal period
        
    Returns:
        Complete evaluation results
    """
    calculator = MetricsCalculator(bins, max_lags, seasonal_period)
    
    if y_true.ndim == 1:
        # Univariate case
        metrics = calculator.calculate_all_metrics(y_true, y_pred, y_train)
        return {
            'overall_metrics': metrics,
            'feature_metrics': {'univariate': metrics},
            'aggregate_metrics': {}
        }
    else:
        # Multivariate case
        feature_metrics = calculator.evaluate_multivariate(
            y_true, y_pred, feature_names, y_train
        )
        aggregate_metrics = calculator.calculate_aggregate_metrics(feature_metrics)
        
        return {
            'feature_metrics': feature_metrics,
            'aggregate_metrics': aggregate_metrics
        }


def create_evaluation_report(evaluation_results: Dict, 
                           output_file: Optional[str] = None,
                           title: str = "QGAN Evaluation Report") -> str:
    """
    Create a comprehensive evaluation report.
    
    Args:
        evaluation_results: Results from evaluate_all_metrics
        output_file: Path to save the report (optional)
        title: Report title
        
    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append(f"# {title}")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Feature-wise metrics
    if 'feature_metrics' in evaluation_results:
        report_lines.append("## Feature-wise Metrics")
        report_lines.append("")
        
        for feature_name, metrics in evaluation_results['feature_metrics'].items():
            report_lines.append(f"### {feature_name}")
            report_lines.append("")
            
            # Group metrics by category
            basic_metrics = {k: v for k, v in metrics.items() 
                           if k in ['mae', 'rmse', 'mse', 'mape', 'smape', 'mbe', 'r_squared']}
            
            if basic_metrics:
                report_lines.append("**Basic Error Metrics:**")
                for metric, value in basic_metrics.items():
                    report_lines.append(f"- {metric.upper()}: {value:.6f}")
                report_lines.append("")
            
            # Statistical metrics
            stat_metrics = {k: v for k, v in metrics.items() 
                          if k in ['pearson_correlation', 'spearman_correlation', 
                                 'wasserstein_distance', 'kl_divergence', 'js_divergence']}
            
            if stat_metrics:
                report_lines.append("**Statistical Metrics:**")
                for metric, value in stat_metrics.items():
                    report_lines.append(f"- {metric.replace('_', ' ').title()}: {value:.6f}")
                report_lines.append("")
            
            # Time series metrics
            ts_metrics = {k: v for k, v in metrics.items() 
                        if k in ['dtw_distance', 'autocorrelation_distance', 
                               'trend_similarity', 'seasonality_similarity']}
            
            if ts_metrics:
                report_lines.append("**Time Series Metrics:**")
                for metric, value in ts_metrics.items():
                    report_lines.append(f"- {metric.replace('_', ' ').title()}: {value:.6f}")
                report_lines.append("")
    
    # Aggregate metrics
    if 'aggregate_metrics' in evaluation_results and evaluation_results['aggregate_metrics']:
        report_lines.append("## Aggregate Metrics (across all features)")
        report_lines.append("")
        
        aggregate = evaluation_results['aggregate_metrics']
        
        # Key aggregate metrics
        key_metrics = ['mae_mean', 'rmse_mean', 'pearson_correlation_mean', 'wasserstein_distance_mean']
        
        for metric in key_metrics:
            if metric in aggregate:
                std_metric = metric.replace('_mean', '_std')
                std_val = aggregate.get(std_metric, 0)
                report_lines.append(f"- {metric.replace('_', ' ').title()}: {aggregate[metric]:.6f} ± {std_val:.6f}")
        
        report_lines.append("")
    
    # Overall assessment
    report_lines.append("## Overall Assessment")
    report_lines.append("")
    
    # Simple scoring system
    if 'feature_metrics' in evaluation_results:
        all_metrics = evaluation_results['feature_metrics']
        if len(all_metrics) == 1:
            metrics = list(all_metrics.values())[0]
        else:
            metrics = evaluation_results.get('aggregate_metrics', {})
            # Use mean values for assessment
            metrics = {k.replace('_mean', ''): v for k, v in metrics.items() if k.endswith('_mean')}
        
        score_components = []
        
        # Error-based scoring (lower is better)
        if 'mae' in metrics:
            score_components.append(f"MAE: {metrics['mae']:.4f}")
        if 'rmse' in metrics:
            score_components.append(f"RMSE: {metrics['rmse']:.4f}")
        
        # Correlation-based scoring (higher is better)  
        if 'pearson_correlation' in metrics:
            corr = metrics['pearson_correlation']
            score_components.append(f"Correlation: {corr:.4f}")
        
        # Distribution similarity (lower distance is better)
        if 'wasserstein_distance' in metrics:
            wd = metrics['wasserstein_distance']
            score_components.append(f"Wasserstein Distance: {wd:.4f}")
        
        if score_components:
            report_lines.append("**Key Performance Indicators:**")
            for component in score_components:
                report_lines.append(f"- {component}")
        
        report_lines.append("")
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to {output_file}")
    
    return report_text


def compare_distributions(y_true: np.ndarray, y_pred: np.ndarray,
                         feature_idx: int = 0, bins: int = 50,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Visualize distribution comparison between real and generated data.
    
    Args:
        y_true: True data
        y_pred: Generated data
        feature_idx: Feature index for multivariate data
        bins: Number of histogram bins
        figsize: Figure size
    """
    if y_true.ndim > 1:
        y_true = y_true[:, feature_idx]
        y_pred = y_pred[:, feature_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Distribution Comparison - Feature {feature_idx}', fontsize=16)
    
    # Histograms
    axes[0, 0].hist(y_true, bins=bins, alpha=0.7, label='Real', density=True)
    axes[0, 0].hist(y_pred, bins=bins, alpha=0.7, label='Generated', density=True)
    axes[0, 0].set_title('Probability Distributions')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    real_quantiles = np.percentile(y_true, np.linspace(0, 100, 100))
    gen_quantiles = np.percentile(y_pred, np.linspace(0, 100, 100))
    axes[0, 1].scatter(real_quantiles, gen_quantiles, alpha=0.6)
    axes[0, 1].plot([min(real_quantiles), max(real_quantiles)], 
                    [min(real_quantiles), max(real_quantiles)], 'r--', lw=2)
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].set_xlabel('Real Data Quantiles')
    axes[0, 1].set_ylabel('Generated Data Quantiles')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Time series overlay
    n_samples = min(500, len(y_true), len(y_pred))
    time_idx = np.arange(n_samples)
    axes[1, 0].plot(time_idx, y_true[:n_samples], alpha=0.7, label='Real', linewidth=1)
    axes[1, 0].plot(time_idx, y_pred[:n_samples], alpha=0.7, label='Generated', linewidth=1)
    axes[1, 0].set_title('Time Series Overlay')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plots
    data_to_plot = [y_true, y_pred]
    bp = axes[1, 1].boxplot(data_to_plot, labels=['Real', 'Generated'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_title('Box Plot Comparison')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_metrics_heatmap(evaluation_results: Dict, 
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a heatmap of metrics across features.
    
    Args:
        evaluation_results: Results from evaluate_all_metrics
        figsize: Figure size
    """
    if 'feature_metrics' not in evaluation_results:
        print("No feature metrics available for heatmap")
        return
    
    feature_metrics = evaluation_results['feature_metrics']
    
    # Create DataFrame
    metrics_df = pd.DataFrame(feature_metrics).T
    
    # Select key metrics for visualization
    key_metrics = ['mae', 'rmse', 'pearson_correlation', 'wasserstein_distance', 
                  'dtw_distance', 'trend_similarity']
    
    available_metrics = [m for m in key_metrics if m in metrics_df.columns]
    
    if not available_metrics:
        print("No key metrics available for heatmap")
        return
    
    metrics_subset = metrics_df[available_metrics]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(metrics_subset, annot=True, cmap='RdYlBu_r', center=0, 
                fmt='.4f', cbar_kws={'label': 'Metric Value'})
    plt.title('Metrics Heatmap Across Features')
    plt.xlabel('Metrics')
    plt.ylabel('Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage function
def quick_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                    feature_names: Optional[List[str]] = None,
                    plot: bool = True, report: bool = True) -> Dict:
    """
    Quick evaluation with default settings.
    
    Args:
        y_true: True data
        y_pred: Generated data
        feature_names: Feature names
        plot: Whether to create plots
        report: Whether to print report
        
    Returns:
        Evaluation results
    """
    # Calculate metrics
    results = evaluate_all_metrics(y_true, y_pred, feature_names=feature_names)
    
    # Create report
    if report:
        report_text = create_evaluation_report(results)
        print(report_text)
    
    # Create plots
    if plot:
        if y_true.ndim > 1:
            # Multivariate: plot first feature and heatmap
            compare_distributions(y_true, y_pred, feature_idx=0)
            plot_metrics_heatmap(results)
        else:
            # Univariate: just distribution comparison
            compare_distributions(y_true, y_pred)
    
    return results 