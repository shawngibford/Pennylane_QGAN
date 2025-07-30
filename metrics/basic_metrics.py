"""
Basic regression and error metrics for time series evaluation.

This module contains fundamental metrics commonly used in regression
and time series forecasting to measure the accuracy of generated data.
"""

import numpy as np
import warnings
from typing import Union, Optional


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE value as percentage
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Add epsilon to avoid division by zero
        denominator = np.abs(y_true) + epsilon
        ape = np.abs((y_true - y_pred) / denominator) * 100
        return np.mean(ape)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        sMAPE value as percentage
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(numerator / denominator) * 100


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MBE value
    """
    return np.mean(y_pred - y_true)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Maximum absolute error
    """
    return np.max(np.abs(y_true - y_pred))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Median absolute error
    """
    return np.median(np.abs(y_true - y_pred))


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray, 
                   normalization: str = 'range') -> float:
    """
    Normalized Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        normalization: Type of normalization ('range', 'mean', 'std')
        
    Returns:
        Normalized RMSE value
    """
    rmse_val = rmse(y_true, y_pred)
    
    if normalization == 'range':
        norm_factor = np.max(y_true) - np.min(y_true)
    elif normalization == 'mean':
        norm_factor = np.mean(y_true)
    elif normalization == 'std':
        norm_factor = np.std(y_true)
    else:
        raise ValueError("normalization must be 'range', 'mean', or 'std'")
    
    if norm_factor == 0:
        return np.inf
    
    return rmse_val / norm_factor


def mase(y_true: np.ndarray, y_pred: np.ndarray, 
         y_train: Optional[np.ndarray] = None, seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling (if None, uses y_true)
        seasonality: Seasonal period for naive forecast
        
    Returns:
        MASE value
    """
    if y_train is None:
        y_train = y_true
    
    # Calculate naive forecast error (seasonal naive)
    if len(y_train) <= seasonality:
        # Use simple naive forecast if not enough data for seasonal
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_forecast = y_train[:-seasonality]
        naive_actual = y_train[seasonality:]
        naive_errors = np.abs(naive_actual - naive_forecast)
    
    scale = np.mean(naive_errors)
    
    if scale == 0:
        return np.inf
    
    return mae(y_true, y_pred) / scale


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (Coefficient of Determination).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Explained Variance Score.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Explained variance score
    """
    y_true_mean = np.mean(y_true)
    numerator = np.var(y_true - y_pred)
    denominator = np.var(y_true)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    
    return 1 - (numerator / denominator)


def mean_absolute_scaled_error_multivariate(y_true: np.ndarray, y_pred: np.ndarray,
                                          y_train: Optional[np.ndarray] = None) -> np.ndarray:
    """
    MASE for multivariate time series (feature-wise).
    
    Args:
        y_true: True values (n_samples, n_features)
        y_pred: Predicted values (n_samples, n_features)
        y_train: Training data for scaling
        
    Returns:
        MASE values for each feature
    """
    if y_true.ndim == 1:
        return np.array([mase(y_true, y_pred, y_train)])
    
    mase_values = []
    for i in range(y_true.shape[1]):
        train_feature = y_train[:, i] if y_train is not None else None
        mase_val = mase(y_true[:, i], y_pred[:, i], train_feature)
        mase_values.append(mase_val)
    
    return np.array(mase_values)


def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_train: Optional[np.ndarray] = None) -> dict:
    """
    Calculate all basic metrics at once.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data (for MASE calculation)
        
    Returns:
        Dictionary with all basic metric values
    """
    metrics = {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mse': mse(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'smape': smape(y_true, y_pred),
        'mbe': mbe(y_true, y_pred),
        'max_error': max_error(y_true, y_pred),
        'median_absolute_error': median_absolute_error(y_true, y_pred),
        'normalized_rmse_range': normalized_rmse(y_true, y_pred, 'range'),
        'normalized_rmse_mean': normalized_rmse(y_true, y_pred, 'mean'),
        'r_squared': r_squared(y_true, y_pred),
        'explained_variance': explained_variance(y_true, y_pred)
    }
    
    # Add MASE if training data is provided
    if y_train is not None:
        metrics['mase'] = mase(y_true, y_pred, y_train)
    
    return metrics 