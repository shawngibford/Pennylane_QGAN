"""
Statistical and distributional metrics for time series evaluation.

This module contains statistical measures and distributional comparison
metrics for evaluating how well generated data matches real data distributions.
"""

import numpy as np
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance as scipy_wasserstein
from typing import Tuple, Optional, Union


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Pearson correlation coefficient
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    correlation, _ = stats.pearsonr(y_true.flatten(), y_pred.flatten())
    return correlation if not np.isnan(correlation) else 0.0


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Spearman correlation coefficient
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    
    correlation, _ = stats.spearmanr(y_true.flatten(), y_pred.flatten())
    return correlation if not np.isnan(correlation) else 0.0


def wasserstein_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Wasserstein (Earth Mover's) distance between distributions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Wasserstein distance
    """
    try:
        return scipy_wasserstein(y_true.flatten(), y_pred.flatten())
    except Exception:
        return np.inf


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 50) -> float:
    """
    Kullback-Leibler divergence between distributions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        bins: Number of bins for histogram estimation
        
    Returns:
        KL divergence (in nats)
    """
    # Create common bin edges
    combined_data = np.concatenate([y_true.flatten(), y_pred.flatten()])
    bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)
    
    # Calculate histograms
    hist_true, _ = np.histogram(y_true.flatten(), bins=bin_edges, density=True)
    hist_pred, _ = np.histogram(y_pred.flatten(), bins=bin_edges, density=True)
    
    # Normalize to probabilities
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_true = hist_true + epsilon
    hist_pred = hist_pred + epsilon
    
    # Calculate KL divergence
    kl_div = np.sum(hist_true * np.log(hist_true / hist_pred))
    
    return kl_div if not np.isnan(kl_div) and not np.isinf(kl_div) else np.inf


def js_divergence(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 50) -> float:
    """
    Jensen-Shannon divergence between distributions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        bins: Number of bins for histogram estimation
        
    Returns:
        JS divergence
    """
    # Create common bin edges
    combined_data = np.concatenate([y_true.flatten(), y_pred.flatten()])
    bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)
    
    # Calculate histograms
    hist_true, _ = np.histogram(y_true.flatten(), bins=bin_edges, density=True)
    hist_pred, _ = np.histogram(y_pred.flatten(), bins=bin_edges, density=True)
    
    # Normalize to probabilities
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)
    
    # Calculate JS divergence
    js_div = jensenshannon(hist_true, hist_pred) ** 2
    
    return js_div if not np.isnan(js_div) else 1.0


def ks_test_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test statistic and p-value.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Tuple of (KS statistic, p-value)
    """
    try:
        ks_stat, p_value = stats.ks_2samp(y_true.flatten(), y_pred.flatten())
        return ks_stat, p_value
    except Exception:
        return 1.0, 0.0


def anderson_darling_test(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Anderson-Darling test statistic for distribution comparison.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        AD test statistic
    """
    try:
        # Combine and sort data
        combined_data = np.concatenate([y_true.flatten(), y_pred.flatten()])
        combined_sorted = np.sort(combined_data)
        
        # Calculate empirical CDFs
        n1, n2 = len(y_true.flatten()), len(y_pred.flatten())
        
        # Calculate AD statistic approximation
        y_true_sorted = np.sort(y_true.flatten())
        y_pred_sorted = np.sort(y_pred.flatten())
        
        # Use a simplified version of the AD test
        cdf_true = np.searchsorted(y_true_sorted, combined_sorted, side='right') / n1
        cdf_pred = np.searchsorted(y_pred_sorted, combined_sorted, side='right') / n2
        
        ad_stat = np.mean((cdf_true - cdf_pred) ** 2)
        
        return ad_stat
    except Exception:
        return np.inf


def energy_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Energy distance between two distributions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Energy distance
    """
    try:
        n, m = len(y_true.flatten()), len(y_pred.flatten())
        
        # Calculate pairwise distances
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # E[|X-Y|] - 0.5*E[|X-X'|] - 0.5*E[|Y-Y'|]
        xy_dist = np.mean([np.abs(x - y) for x in y_true_flat for y in y_pred_flat])
        xx_dist = np.mean([np.abs(x1 - x2) for x1 in y_true_flat for x2 in y_true_flat])
        yy_dist = np.mean([np.abs(y1 - y2) for y1 in y_pred_flat for y2 in y_pred_flat])
        
        energy_dist = 2 * xy_dist - xx_dist - yy_dist
        
        return energy_dist
    except Exception:
        return np.inf


def maximum_mean_discrepancy(y_true: np.ndarray, y_pred: np.ndarray, 
                           gamma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        gamma: RBF kernel parameter
        
    Returns:
        MMD value
    """
    def rbf_kernel(x, y, gamma):
        return np.exp(-gamma * np.sum((x - y) ** 2))
    
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        n, m = len(y_true_flat), len(y_pred_flat)
        
        # Calculate kernel matrices
        kxx = np.mean([rbf_kernel(x1, x2, gamma) for x1 in y_true_flat for x2 in y_true_flat])
        kyy = np.mean([rbf_kernel(y1, y2, gamma) for y1 in y_pred_flat for y2 in y_pred_flat])
        kxy = np.mean([rbf_kernel(x, y, gamma) for x in y_true_flat for y in y_pred_flat])
        
        mmd = kxx + kyy - 2 * kxy
        
        return max(0, mmd)  # MMD should be non-negative
    except Exception:
        return np.inf


def distribution_overlap(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 50) -> float:
    """
    Overlap coefficient between two distributions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        bins: Number of bins for histogram estimation
        
    Returns:
        Overlap coefficient (0 to 1)
    """
    # Create common bin edges
    combined_data = np.concatenate([y_true.flatten(), y_pred.flatten()])
    bin_edges = np.linspace(combined_data.min(), combined_data.max(), bins + 1)
    
    # Calculate histograms
    hist_true, _ = np.histogram(y_true.flatten(), bins=bin_edges, density=True)
    hist_pred, _ = np.histogram(y_pred.flatten(), bins=bin_edges, density=True)
    
    # Normalize to probabilities
    hist_true = hist_true / np.sum(hist_true)
    hist_pred = hist_pred / np.sum(hist_pred)
    
    # Calculate overlap (sum of minimum probabilities)
    overlap = np.sum(np.minimum(hist_true, hist_pred))
    
    return overlap


def calculate_statistical_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                bins: int = 50) -> dict:
    """
    Calculate all statistical metrics at once.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        bins: Number of bins for histogram-based metrics
        
    Returns:
        Dictionary with all statistical metric values
    """
    # Basic correlations
    pearson_corr = pearson_correlation(y_true, y_pred)
    spearman_corr = spearman_correlation(y_true, y_pred)
    
    # Distributional metrics
    wasserstein_dist = wasserstein_distance(y_true, y_pred)
    kl_div = kl_divergence(y_true, y_pred, bins)
    js_div = js_divergence(y_true, y_pred, bins)
    
    # Statistical tests
    ks_stat, ks_pvalue = ks_test_statistic(y_true, y_pred)
    ad_stat = anderson_darling_test(y_true, y_pred)
    
    # Advanced metrics
    energy_dist = energy_distance(y_true, y_pred)
    mmd = maximum_mean_discrepancy(y_true, y_pred)
    overlap = distribution_overlap(y_true, y_pred, bins)
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'wasserstein_distance': wasserstein_dist,
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'anderson_darling_statistic': ad_stat,
        'energy_distance': energy_dist,
        'maximum_mean_discrepancy': mmd,
        'distribution_overlap': overlap
    } 