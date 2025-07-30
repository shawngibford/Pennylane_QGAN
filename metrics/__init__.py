"""
Comprehensive metrics package for evaluating QGAN performance.

This package provides 20+ common evaluation metrics for comparing
generated time series data with real data, including:
- Basic regression metrics (MAE, RMSE, MAPE, etc.)
- Statistical measures (correlations, distributions)
- Time series specific metrics (DTW, autocorrelation)
- Generative model metrics (Wasserstein, KL divergence)
"""

from .basic_metrics import (
    mae, rmse, mse, mape, smape, mbe, max_error, median_absolute_error,
    normalized_rmse, mase, r_squared, explained_variance
)

from .statistical_metrics import (
    pearson_correlation, spearman_correlation, 
    wasserstein_distance, kl_divergence, js_divergence,
    ks_test_statistic, anderson_darling_test
)

from .time_series_metrics import (
    dtw_distance, autocorrelation_distance, frechet_distance,
    trend_similarity, seasonality_similarity
)

from .comprehensive_evaluation import (
    evaluate_all_metrics, create_evaluation_report,
    compare_distributions, plot_metrics_heatmap, MetricsCalculator
)

__version__ = "1.0.0"
__author__ = "QGAN Evaluation Suite"

# Main metrics list for easy access
AVAILABLE_METRICS = [
    'mae', 'rmse', 'mse', 'mape', 'smape', 'mbe', 'max_error', 
    'median_absolute_error', 'normalized_rmse', 'mase', 'r_squared',
    'explained_variance', 'pearson_correlation', 'spearman_correlation',
    'wasserstein_distance', 'kl_divergence', 'js_divergence', 
    'ks_test_statistic', 'dtw_distance', 'autocorrelation_distance'
] 