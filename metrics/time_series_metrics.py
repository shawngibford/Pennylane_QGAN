"""
Time series specific metrics for temporal pattern evaluation.

This module contains metrics specifically designed for time series data,
including temporal distance measures, autocorrelation analysis, and
pattern similarity metrics.
"""

import numpy as np
import warnings
from scipy.spatial.distance import euclidean
from scipy import signal
from typing import Tuple, Optional, List
import statsmodels.api as sm


def dtw_distance(y_true: np.ndarray, y_pred: np.ndarray, 
                window: Optional[int] = None) -> float:
    """
    Dynamic Time Warping distance between two time series.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        window: Constraint window size (None for unconstrained)
        
    Returns:
        DTW distance
    """
    def dtw_core(x, y, window_size=None):
        n, m = len(x), len(y)
        
        # Initialize cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Apply window constraint if specified
        for i in range(1, n + 1):
            start_j = max(1, i - window_size if window_size else 1)
            end_j = min(m + 1, i + window_size + 1 if window_size else m + 1)
            
            for j in range(start_j, end_j):
                cost = (x[i-1] - y[j-1]) ** 2
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        return np.sqrt(dtw_matrix[n, m])
    
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        return dtw_core(y_true_flat, y_pred_flat, window)
    except Exception:
        return np.inf


def autocorrelation_distance(y_true: np.ndarray, y_pred: np.ndarray, 
                           max_lags: int = 20) -> float:
    """
    Distance between autocorrelation functions.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        max_lags: Maximum number of lags to consider
        
    Returns:
        Euclidean distance between autocorrelation functions
    """
    try:
        # Calculate autocorrelations
        acf_true = sm.tsa.acf(y_true.flatten(), nlags=max_lags, fft=True)
        acf_pred = sm.tsa.acf(y_pred.flatten(), nlags=max_lags, fft=True)
        
        # Calculate distance (excluding lag 0 which is always 1)
        acf_distance = np.sqrt(np.sum((acf_true[1:] - acf_pred[1:]) ** 2))
        
        return acf_distance
    except Exception:
        return np.inf


def frechet_distance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fréchet distance between two curves (simplified 1D version).
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        
    Returns:
        Fréchet distance
    """
    def frechet_recursive(p, q, i, j, memo):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == 0 and j == 0:
            result = np.abs(p[0] - q[0])
        elif i > 0 and j == 0:
            result = max(frechet_recursive(p, q, i-1, 0, memo), np.abs(p[i] - q[0]))
        elif i == 0 and j > 0:
            result = max(frechet_recursive(p, q, 0, j-1, memo), np.abs(p[0] - q[j]))
        else:
            result = max(
                min(
                    frechet_recursive(p, q, i-1, j, memo),
                    frechet_recursive(p, q, i-1, j-1, memo),
                    frechet_recursive(p, q, i, j-1, memo)
                ),
                np.abs(p[i] - q[j])
            )
        
        memo[(i, j)] = result
        return result
    
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Downsample if series are too long (for computational efficiency)
        max_length = 100
        if len(y_true_flat) > max_length:
            step = len(y_true_flat) // max_length
            y_true_flat = y_true_flat[::step]
            y_pred_flat = y_pred_flat[::step]
        
        memo = {}
        return frechet_recursive(y_true_flat, y_pred_flat, 
                               len(y_true_flat)-1, len(y_pred_flat)-1, memo)
    except Exception:
        return np.inf


def trend_similarity(y_true: np.ndarray, y_pred: np.ndarray, 
                    method: str = 'linear') -> float:
    """
    Similarity between trends in two time series.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        method: Trend extraction method ('linear', 'polynomial')
        
    Returns:
        Trend similarity score (higher is better)
    """
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        x = np.arange(len(y_true_flat))
        
        if method == 'linear':
            # Linear trend
            trend_true = np.polyfit(x, y_true_flat, 1)
            trend_pred = np.polyfit(x, y_pred_flat, 1)
            
            # Compare slopes and intercepts
            slope_diff = abs(trend_true[0] - trend_pred[0])
            intercept_diff = abs(trend_true[1] - trend_pred[1])
            
            # Normalize by data range
            data_range = max(np.ptp(y_true_flat), np.ptp(y_pred_flat))
            if data_range == 0:
                return 1.0
            
            similarity = 1 / (1 + slope_diff + intercept_diff / data_range)
            
        else:  # polynomial
            # Quadratic trend
            trend_true = np.polyfit(x, y_true_flat, 2)
            trend_pred = np.polyfit(x, y_pred_flat, 2)
            
            # Compare coefficients
            coeff_diff = np.sum(np.abs(trend_true - trend_pred))
            similarity = 1 / (1 + coeff_diff)
        
        return similarity
    except Exception:
        return 0.0


def seasonality_similarity(y_true: np.ndarray, y_pred: np.ndarray,
                         period: Optional[int] = None) -> float:
    """
    Similarity between seasonal patterns in two time series.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        period: Expected seasonal period (auto-detected if None)
        
    Returns:
        Seasonality similarity score
    """
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        if period is None:
            # Simple period detection using FFT
            fft_true = np.fft.fft(y_true_flat)
            freqs = np.fft.fftfreq(len(y_true_flat))
            # Find dominant frequency (excluding DC component)
            dominant_freq_idx = np.argmax(np.abs(fft_true[1:len(fft_true)//2])) + 1
            period = int(1 / abs(freqs[dominant_freq_idx])) if freqs[dominant_freq_idx] != 0 else len(y_true_flat) // 4
        
        if period <= 1 or period >= len(y_true_flat) // 2:
            return 0.0
        
        # Extract seasonal components using simple decomposition
        def extract_seasonal(series, period):
            n_periods = len(series) // period
            if n_periods < 2:
                return series
            
            # Reshape and average across periods
            truncated = series[:n_periods * period]
            reshaped = truncated.reshape(n_periods, period)
            seasonal = np.mean(reshaped, axis=0)
            
            return seasonal
        
        seasonal_true = extract_seasonal(y_true_flat, period)
        seasonal_pred = extract_seasonal(y_pred_flat, period)
        
        # Calculate correlation between seasonal patterns
        if len(seasonal_true) > 1 and len(seasonal_pred) > 1:
            correlation = np.corrcoef(seasonal_true, seasonal_pred)[0, 1]
            similarity = max(0, correlation)  # Only positive correlations
        else:
            similarity = 0.0
        
        return similarity if not np.isnan(similarity) else 0.0
    except Exception:
        return 0.0


def volatility_similarity(y_true: np.ndarray, y_pred: np.ndarray,
                         window: int = 10) -> float:
    """
    Similarity between volatility patterns (rolling standard deviation).
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        window: Rolling window size
        
    Returns:
        Volatility similarity score
    """
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        if len(y_true_flat) < window:
            return 0.0
        
        # Calculate rolling volatility
        vol_true = []
        vol_pred = []
        
        for i in range(window, len(y_true_flat)):
            vol_true.append(np.std(y_true_flat[i-window:i]))
            vol_pred.append(np.std(y_pred_flat[i-window:i]))
        
        vol_true = np.array(vol_true)
        vol_pred = np.array(vol_pred)
        
        # Calculate correlation between volatilities
        if len(vol_true) > 1:
            correlation = np.corrcoef(vol_true, vol_pred)[0, 1]
            similarity = max(0, correlation)
        else:
            similarity = 0.0
        
        return similarity if not np.isnan(similarity) else 0.0
    except Exception:
        return 0.0


def spectral_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Similarity between power spectral densities.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        
    Returns:
        Spectral similarity score
    """
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Calculate power spectral densities
        freqs_true, psd_true = signal.welch(y_true_flat)
        freqs_pred, psd_pred = signal.welch(y_pred_flat)
        
        # Normalize PSDs
        psd_true = psd_true / np.sum(psd_true)
        psd_pred = psd_pred / np.sum(psd_pred)
        
        # Calculate similarity (1 - JS divergence)
        # Use Jensen-Shannon divergence
        m = (psd_true + psd_pred) / 2
        js_div = 0.5 * np.sum(psd_true * np.log2(psd_true / m + 1e-10)) + \
                 0.5 * np.sum(psd_pred * np.log2(psd_pred / m + 1e-10))
        
        similarity = 1 - js_div
        
        return max(0, similarity)
    except Exception:
        return 0.0


def phase_space_similarity(y_true: np.ndarray, y_pred: np.ndarray,
                          embedding_dim: int = 3, delay: int = 1) -> float:
    """
    Similarity between phase space reconstructions.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        embedding_dim: Embedding dimension
        delay: Time delay
        
    Returns:
        Phase space similarity score
    """
    def embed_series(series, dim, delay):
        n = len(series) - (dim - 1) * delay
        if n <= 0:
            return np.array([])
        
        embedded = np.zeros((n, dim))
        for i in range(dim):
            embedded[:, i] = series[i * delay:i * delay + n]
        
        return embedded
    
    try:
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Create phase space embeddings
        embed_true = embed_series(y_true_flat, embedding_dim, delay)
        embed_pred = embed_series(y_pred_flat, embedding_dim, delay)
        
        if len(embed_true) == 0 or len(embed_pred) == 0:
            return 0.0
        
        # Calculate mean distances in phase space
        # Use a subset for computational efficiency
        max_points = 100
        if len(embed_true) > max_points:
            idx = np.random.choice(len(embed_true), max_points, replace=False)
            embed_true = embed_true[idx]
        if len(embed_pred) > max_points:
            idx = np.random.choice(len(embed_pred), max_points, replace=False)
            embed_pred = embed_pred[idx]
        
        # Calculate average distance between embeddings
        distances = []
        for i, point_true in enumerate(embed_true):
            if i < len(embed_pred):
                dist = euclidean(point_true, embed_pred[i])
                distances.append(dist)
        
        if distances:
            avg_distance = np.mean(distances)
            # Convert to similarity score
            similarity = 1 / (1 + avg_distance)
        else:
            similarity = 0.0
        
        return similarity
    except Exception:
        return 0.0


def calculate_time_series_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                max_lags: int = 20, period: Optional[int] = None) -> dict:
    """
    Calculate all time series specific metrics at once.
    
    Args:
        y_true: True time series
        y_pred: Predicted time series
        max_lags: Maximum lags for autocorrelation
        period: Seasonal period for seasonality analysis
        
    Returns:
        Dictionary with all time series metric values
    """
    return {
        'dtw_distance': dtw_distance(y_true, y_pred),
        'autocorrelation_distance': autocorrelation_distance(y_true, y_pred, max_lags),
        'frechet_distance': frechet_distance(y_true, y_pred),
        'trend_similarity': trend_similarity(y_true, y_pred),
        'seasonality_similarity': seasonality_similarity(y_true, y_pred, period),
        'volatility_similarity': volatility_similarity(y_true, y_pred),
        'spectral_similarity': spectral_similarity(y_true, y_pred),
        'phase_space_similarity': phase_space_similarity(y_true, y_pred)
    } 