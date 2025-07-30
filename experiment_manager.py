"""
Comprehensive Experiment Management System for QGAN Research.

This module provides a complete framework for managing, tracking, and reporting
on QGAN experiments. Each experiment gets its own folder with all artifacts,
plots, metrics, and detailed reports.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import warnings
import shutil
from pathlib import Path

from benchmark_dataset import create_benchmark_dataset
from metrics import evaluate_all_metrics, create_evaluation_report, compare_distributions, plot_metrics_heatmap


class ExperimentManager:
    """
    Comprehensive experiment management and tracking system.
    """
    
    def __init__(self, base_experiment_dir: str = "experiments"):
        """
        Initialize the experiment manager.
        
        Args:
            base_experiment_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_experiment_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create main experiment log
        self.experiment_log_file = self.base_dir / "experiment_log.json"
        self.experiment_log = self.load_experiment_log()
    
    def load_experiment_log(self) -> Dict:
        """Load the experiment log or create a new one."""
        if self.experiment_log_file.exists():
            with open(self.experiment_log_file, 'r') as f:
                return json.load(f)
        return {"experiments": [], "last_experiment_id": 0}
    
    def save_experiment_log(self):
        """Save the experiment log."""
        with open(self.experiment_log_file, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)
    
    def create_experiment(self, experiment_name: str, description: str = "",
                         tags: List[str] = None) -> str:
        """
        Create a new experiment with unique ID and folder structure.
        
        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            tags: List of tags for categorization
            
        Returns:
            Experiment ID
        """
        # Generate unique experiment ID
        exp_id = self.experiment_log["last_experiment_id"] + 1
        self.experiment_log["last_experiment_id"] = exp_id
        
        # Create experiment folder structure
        exp_folder = self.base_dir / f"exp_{exp_id:04d}_{experiment_name}"
        exp_folder.mkdir(exist_ok=True)
        
        # Create subfolders
        subfolders = ["data", "models", "plots", "reports", "metrics", "config"]
        for subfolder in subfolders:
            (exp_folder / subfolder).mkdir(exist_ok=True)
        
        # Create experiment metadata
        experiment_metadata = {
            "experiment_id": exp_id,
            "experiment_name": experiment_name,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "folder_path": str(exp_folder),
            "status": "created",
            "results": {}
        }
        
        # Add to experiment log
        self.experiment_log["experiments"].append(experiment_metadata)
        self.save_experiment_log()
        
        # Save experiment metadata in folder
        with open(exp_folder / "experiment_metadata.json", 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        print(f"✅ Created experiment: {exp_id:04d} - {experiment_name}")
        print(f"📁 Experiment folder: {exp_folder}")
        
        return f"{exp_id:04d}"
    
    def run_benchmark_experiment(self, experiment_id: str, 
                                dataset_config: Dict = None,
                                model_config: Dict = None) -> Dict:
        """
        Run a complete benchmark experiment with dataset generation and evaluation.
        
        Args:
            experiment_id: Experiment ID
            dataset_config: Configuration for dataset generation
            model_config: Configuration for model (placeholder for future)
            
        Returns:
            Experiment results
        """
        exp_folder = self.get_experiment_folder(experiment_id)
        if not exp_folder:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        print(f"🚀 Running benchmark experiment {experiment_id}...")
        
        # Default dataset configuration
        default_dataset_config = {
            "length": 2000,
            "window_size": 50,
            "normalize": True,
            "add_regime_changes": False
        }
        dataset_config = {**default_dataset_config, **(dataset_config or {})}
        
        # Generate benchmark dataset
        print("📊 Generating benchmark dataset...")
        dataset = create_benchmark_dataset(
            **dataset_config,
            save_csv=str(exp_folder / "data" / "sin_data.csv"),
            save_path=str(exp_folder / "data" / "benchmark_dataset.npz")
        )
        
        # Save dataset configuration
        with open(exp_folder / "config" / "dataset_config.json", 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        # Generate synthetic "QGAN" output for testing (replace with actual model)
        print("🤖 Generating synthetic QGAN output (placeholder)...")
        real_data = dataset['raw_data']
        
        # Create realistic but imperfect synthetic data for testing
        synthetic_data = self.generate_test_synthetic_data(real_data)
        
        # Save synthetic data
        np.save(exp_folder / "data" / "synthetic_qgan_output.npy", synthetic_data)
        
        # Run comprehensive evaluation
        print("📈 Running comprehensive evaluation...")
        results = self.evaluate_experiment(
            experiment_id=experiment_id,
            real_data=real_data,
            generated_data=synthetic_data,
            feature_names=['trend', 'seasonal', 'detailed', 'complex', 'correlated']
        )
        
        print(f"✅ Experiment {experiment_id} completed successfully!")
        
        return results
    
    def run_real_data_experiment(self, experiment_id: str,
                               csv_file_path: str,
                               feature_columns: List[str] = None,
                               target_features: int = 5,
                               normalize: bool = True) -> Dict:
        """
        Run a complete experiment on real CSV data.
        
        Args:
            experiment_id: Experiment ID
            csv_file_path: Path to the CSV file
            feature_columns: List of column names to use as features
            target_features: Number of features to select if feature_columns is None
            normalize: Whether to normalize the data
            
        Returns:
            Experiment results
        """
        exp_folder = self.get_experiment_folder(experiment_id)
        if not exp_folder:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        print(f"🚀 Running real data experiment {experiment_id}...")
        
        # Load and process CSV data
        print(f"📊 Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        
        # Auto-detect numeric columns if feature_columns not specified
        if feature_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove any date/time columns that might be numeric
            date_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'min', 'sec']
            numeric_columns = [col for col in numeric_columns 
                             if not any(keyword in col.lower() for keyword in date_keywords)]
            
            # Select the specified number of features
            if len(numeric_columns) > target_features:
                # Prefer columns with more variation (higher std)
                variations = df[numeric_columns].std()
                feature_columns = variations.nlargest(target_features).index.tolist()
            else:
                feature_columns = numeric_columns[:target_features]
        
        print(f"📈 Selected features: {feature_columns}")
        
        # Extract feature data
        real_data = df[feature_columns].values
        
        # Handle missing values
        if np.any(np.isnan(real_data)):
            print("⚠️  Found NaN values, filling with column means...")
            real_data = pd.DataFrame(real_data).fillna(pd.DataFrame(real_data).mean()).values
        
        # Normalize if requested
        if normalize:
            print("🔄 Normalizing data...")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            real_data = scaler.fit_transform(real_data)
            
            # Save scaler for future use
            import pickle
            with open(exp_folder / "config" / "scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save processed data
        processed_df = pd.DataFrame(real_data, columns=feature_columns)
        processed_df.to_csv(exp_folder / "data" / "processed_real_data.csv", index=False)
        
        # Copy original data to experiment folder
        shutil.copy2(csv_file_path, exp_folder / "data" / "original_data.csv")
        
        # Save data processing configuration
        data_config = {
            "original_file": csv_file_path,
            "feature_columns": feature_columns,
            "normalize": normalize,
            "data_shape": real_data.shape,
            "total_samples": len(real_data),
            "features_selected": len(feature_columns)
        }
        
        with open(exp_folder / "config" / "data_config.json", 'w') as f:
            json.dump(data_config, f, indent=2)
        
        # Generate synthetic "QGAN" output for comparison (placeholder)
        print("🤖 Generating synthetic QGAN output (placeholder)...")
        synthetic_data = self.generate_test_synthetic_data(real_data)
        
        # Save synthetic data
        np.save(exp_folder / "data" / "synthetic_qgan_output.npy", synthetic_data)
        
        # Run comprehensive evaluation
        print("📈 Running comprehensive evaluation...")
        results = self.evaluate_experiment(
            experiment_id=experiment_id,
            real_data=real_data,
            generated_data=synthetic_data,
            feature_names=feature_columns
        )
        
        print(f"✅ Real data experiment {experiment_id} completed successfully!")
        print(f"📊 Processed {len(feature_columns)} features from {len(real_data)} samples")
        
        return results
    
    def generate_test_synthetic_data(self, real_data: np.ndarray) -> np.ndarray:
        """
        Generate synthetic test data that mimics imperfect QGAN output.
        This is a placeholder until actual QGAN model is integrated.
        """
        # Add some realistic imperfections to test the evaluation system
        np.random.seed(42)  # For reproducible test results
        
        # Start with real data and add controlled noise/distortions
        synthetic_data = real_data.copy()
        
        # Add some noise
        noise_level = 0.1 * np.std(real_data, axis=0)
        synthetic_data += np.random.normal(0, noise_level, synthetic_data.shape)
        
        # Slightly shift the distribution
        synthetic_data += np.random.normal(0, 0.05, synthetic_data.shape[1])
        
        # Add some temporal distortion to make it more realistic
        for i in range(synthetic_data.shape[1]):
            # Slight frequency shift
            t = np.arange(len(synthetic_data))
            freq_shift = 0.02 * np.sin(0.01 * t)
            synthetic_data[:, i] += freq_shift * np.std(synthetic_data[:, i]) * 0.1
        
        return synthetic_data
    
    def evaluate_experiment(self, experiment_id: str, real_data: np.ndarray,
                          generated_data: np.ndarray, feature_names: List[str] = None) -> Dict:
        """
        Run comprehensive evaluation and generate all plots and reports.
        
        Args:
            experiment_id: Experiment ID
            real_data: Real benchmark data
            generated_data: Generated data from QGAN
            feature_names: Names of features
            
        Returns:
            Evaluation results
        """
        exp_folder = self.get_experiment_folder(experiment_id)
        
        # Run comprehensive metrics evaluation
        print("📊 Calculating all metrics...")
        evaluation_results = evaluate_all_metrics(
            y_true=real_data,
            y_pred=generated_data,
            feature_names=feature_names
        )
        
        # Save metrics results
        metrics_file = exp_folder / "metrics" / "evaluation_results.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types to JSON serializable
            json_results = self.convert_to_json_serializable(evaluation_results)
            json.dump(json_results, f, indent=2)
        
        # Generate all plots
        print("📈 Generating plots...")
        self.generate_all_plots(exp_folder, real_data, generated_data, 
                               evaluation_results, feature_names)
        
        # Generate comprehensive report
        print("📝 Generating comprehensive report...")
        self.generate_comprehensive_report(experiment_id, evaluation_results, 
                                         real_data, generated_data, feature_names)
        
        # Update experiment metadata
        self.update_experiment_status(experiment_id, "completed", evaluation_results)
        
        return evaluation_results
    
    def generate_all_plots(self, exp_folder: Path, real_data: np.ndarray,
                          generated_data: np.ndarray, evaluation_results: Dict,
                          feature_names: List[str] = None):
        """Generate all visualization plots for the experiment."""
        plots_dir = exp_folder / "plots"
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(real_data.shape[1])]
        
        # Set style for consistent plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Distribution comparisons for each feature
        for i, feature_name in enumerate(feature_names):
            plt.figure(figsize=(15, 10))
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Feature Analysis: {feature_name}', fontsize=16)
            
            real_feature = real_data[:, i]
            gen_feature = generated_data[:, i]
            
            # Histogram comparison
            axes[0, 0].hist(real_feature, bins=50, alpha=0.7, label='Real', density=True)
            axes[0, 0].hist(gen_feature, bins=50, alpha=0.7, label='Generated', density=True)
            axes[0, 0].set_title('Distribution Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Time series comparison
            n_samples = min(500, len(real_feature))
            axes[0, 1].plot(real_feature[:n_samples], alpha=0.8, label='Real', linewidth=1)
            axes[0, 1].plot(gen_feature[:n_samples], alpha=0.8, label='Generated', linewidth=1)
            axes[0, 1].set_title('Time Series Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Q-Q plot
            real_quantiles = np.percentile(real_feature, np.linspace(0, 100, 100))
            gen_quantiles = np.percentile(gen_feature, np.linspace(0, 100, 100))
            axes[0, 2].scatter(real_quantiles, gen_quantiles, alpha=0.6)
            axes[0, 2].plot([real_quantiles.min(), real_quantiles.max()], 
                           [real_quantiles.min(), real_quantiles.max()], 'r--', lw=2)
            axes[0, 2].set_title('Q-Q Plot')
            axes[0, 2].set_xlabel('Real Quantiles')
            axes[0, 2].set_ylabel('Generated Quantiles')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Box plot
            data_to_plot = [real_feature, gen_feature]
            bp = axes[1, 0].boxplot(data_to_plot, labels=['Real', 'Generated'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            axes[1, 0].set_title('Distribution Statistics')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Autocorrelation comparison
            try:
                import statsmodels.api as sm
                acf_real = sm.tsa.acf(real_feature, nlags=20, fft=True)
                acf_gen = sm.tsa.acf(gen_feature, nlags=20, fft=True)
                lags = range(len(acf_real))
                axes[1, 1].plot(lags, acf_real, 'o-', label='Real', alpha=0.8)
                axes[1, 1].plot(lags, acf_gen, 's-', label='Generated', alpha=0.8)
                axes[1, 1].set_title('Autocorrelation Comparison')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            except ImportError:
                axes[1, 1].text(0.5, 0.5, 'Statsmodels not available\nfor ACF calculation', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
            
            # Power spectral density
            try:
                from scipy import signal
                freqs_real, psd_real = signal.welch(real_feature)
                freqs_gen, psd_gen = signal.welch(gen_feature)
                axes[1, 2].semilogy(freqs_real, psd_real, label='Real', alpha=0.8)
                axes[1, 2].semilogy(freqs_gen, psd_gen, label='Generated', alpha=0.8)
                axes[1, 2].set_title('Power Spectral Density')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
            except Exception:
                axes[1, 2].text(0.5, 0.5, 'Error calculating PSD', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'feature_analysis_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Overall metrics heatmap
        if 'feature_metrics' in evaluation_results:
            plt.figure(figsize=(14, 8))
            feature_metrics = evaluation_results['feature_metrics']
            metrics_df = pd.DataFrame(feature_metrics).T
            
            # Select key metrics for heatmap
            key_metrics = ['mae', 'rmse', 'pearson_correlation', 'wasserstein_distance', 
                          'dtw_distance', 'trend_similarity']
            available_metrics = [m for m in key_metrics if m in metrics_df.columns]
            
            if available_metrics:
                metrics_subset = metrics_df[available_metrics]
                sns.heatmap(metrics_subset, annot=True, cmap='RdYlBu_r', center=0, 
                           fmt='.4f', cbar_kws={'label': 'Metric Value'})
                plt.title('Metrics Heatmap Across Features')
                plt.xlabel('Metrics')
                plt.ylabel('Features')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Metrics summary bar chart
        if 'aggregate_metrics' in evaluation_results and evaluation_results['aggregate_metrics']:
            plt.figure(figsize=(12, 8))
            aggregate = evaluation_results['aggregate_metrics']
            
            # Select key aggregate metrics
            key_aggregate = ['mae_mean', 'rmse_mean', 'pearson_correlation_mean', 'wasserstein_distance_mean']
            values = []
            labels = []
            errors = []
            
            for metric in key_aggregate:
                if metric in aggregate:
                    values.append(aggregate[metric])
                    labels.append(metric.replace('_mean', '').replace('_', ' ').title())
                    std_metric = metric.replace('_mean', '_std')
                    errors.append(aggregate.get(std_metric, 0))
            
            if values:
                bars = plt.bar(labels, values, yerr=errors, capsize=5, alpha=0.7)
                plt.title('Key Performance Metrics (Mean ± Std)')
                plt.ylabel('Metric Value')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                
                # Color bars based on performance (green=good, red=bad)
                colors = ['green' if 'correlation' in label.lower() else 'red' if any(x in label.lower() for x in ['mae', 'rmse', 'distance']) else 'blue' for label in labels]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    bar.set_alpha(0.7)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"📈 All plots saved to {plots_dir}")
    
    def generate_comprehensive_report(self, experiment_id: str, evaluation_results: Dict,
                                    real_data: np.ndarray, generated_data: np.ndarray,
                                    feature_names: List[str] = None):
        """Generate a comprehensive HTML and Markdown report."""
        exp_folder = self.get_experiment_folder(experiment_id)
        reports_dir = exp_folder / "reports"
        
        # Get experiment metadata
        metadata = self.get_experiment_metadata(experiment_id)
        
        # Generate markdown report
        report_lines = []
        report_lines.append(f"# QGAN Experiment Report: {metadata['experiment_name']}")
        report_lines.append(f"**Experiment ID:** {experiment_id}")
        report_lines.append(f"**Created:** {metadata['created_at']}")
        report_lines.append(f"**Description:** {metadata.get('description', 'No description provided')}")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Dataset information
        report_lines.append("## Dataset Information")
        report_lines.append("")
        report_lines.append(f"- **Data Shape:** {real_data.shape}")
        report_lines.append(f"- **Features:** {feature_names or [f'feature_{i}' for i in range(real_data.shape[1])]}")
        report_lines.append(f"- **Time Series Length:** {real_data.shape[0]}")
        report_lines.append(f"- **Number of Features:** {real_data.shape[1]}")
        report_lines.append("")
        
        # Key results summary
        if 'aggregate_metrics' in evaluation_results and evaluation_results['aggregate_metrics']:
            report_lines.append("## Key Performance Summary")
            report_lines.append("")
            
            aggregate = evaluation_results['aggregate_metrics']
            key_metrics = ['mae_mean', 'rmse_mean', 'pearson_correlation_mean', 'wasserstein_distance_mean']
            
            for metric in key_metrics:
                if metric in aggregate:
                    std_metric = metric.replace('_mean', '_std')
                    std_val = aggregate.get(std_metric, 0)
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}:** {aggregate[metric]:.6f} ± {std_val:.6f}")
            
            report_lines.append("")
        
        # Feature-wise detailed results
        if 'feature_metrics' in evaluation_results:
            report_lines.append("## Detailed Feature Analysis")
            report_lines.append("")
            
            feature_metrics = evaluation_results['feature_metrics']
            
            for feature_name, metrics in feature_metrics.items():
                report_lines.append(f"### {feature_name}")
                report_lines.append("")
                
                # Key metrics for this feature
                key_feature_metrics = ['mae', 'rmse', 'pearson_correlation', 'wasserstein_distance', 'dtw_distance']
                
                for metric in key_feature_metrics:
                    if metric in metrics:
                        report_lines.append(f"- **{metric.upper()}:** {metrics[metric]:.6f}")
                
                report_lines.append("")
        
        # Include plots
        report_lines.append("## Visualizations")
        report_lines.append("")
        
        # List all generated plots
        plots_dir = exp_folder / "plots"
        for plot_file in plots_dir.glob("*.png"):
            relative_path = f"../plots/{plot_file.name}"
            report_lines.append(f"### {plot_file.stem.replace('_', ' ').title()}")
            report_lines.append(f"![{plot_file.stem}]({relative_path})")
            report_lines.append("")
        
        # Conclusions and recommendations
        report_lines.append("## Conclusions and Recommendations")
        report_lines.append("")
        report_lines.append("### Performance Assessment")
        
        # Simple automated assessment
        if 'aggregate_metrics' in evaluation_results and evaluation_results['aggregate_metrics']:
            aggregate = evaluation_results['aggregate_metrics']
            
            # Assess correlation performance
            if 'pearson_correlation_mean' in aggregate:
                corr = aggregate['pearson_correlation_mean']
                if corr > 0.8:
                    report_lines.append("- ✅ **Excellent correlation** between real and generated data")
                elif corr > 0.6:
                    report_lines.append("- ⚠️ **Good correlation** - some room for improvement")
                else:
                    report_lines.append("- ❌ **Poor correlation** - significant improvement needed")
            
            # Assess error metrics
            if 'mae_mean' in aggregate and 'rmse_mean' in aggregate:
                mae_val = aggregate['mae_mean']
                rmse_val = aggregate['rmse_mean']
                report_lines.append(f"- **Error levels:** MAE={mae_val:.4f}, RMSE={rmse_val:.4f}")
        
        report_lines.append("")
        report_lines.append("### Next Steps")
        report_lines.append("- Consider hyperparameter tuning if performance is below expectations")
        report_lines.append("- Analyze feature-specific performance for targeted improvements")
        report_lines.append("- Compare with baseline models for context")
        report_lines.append("")
        
        # Technical details
        report_lines.append("## Technical Details")
        report_lines.append("")
        report_lines.append("### Experiment Configuration")
        
        # Load and include dataset configuration
        config_file = exp_folder / "config" / "dataset_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                dataset_config = json.load(f)
            
            report_lines.append("**Dataset Configuration:**")
            for key, value in dataset_config.items():
                report_lines.append(f"- {key}: {value}")
        
        report_lines.append("")
        report_lines.append("### Files Generated")
        report_lines.append("- Raw data: `data/sin_data.csv`")
        report_lines.append("- Processed dataset: `data/benchmark_dataset.npz`")
        report_lines.append("- QGAN output: `data/synthetic_qgan_output.npy`")
        report_lines.append("- Metrics results: `metrics/evaluation_results.json`")
        report_lines.append("- All plots: `plots/`")
        report_lines.append("")
        
        # Footer
        report_lines.append("---")
        report_lines.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        # Save markdown report
        report_text = "\n".join(report_lines)
        with open(reports_dir / "comprehensive_report.md", 'w') as f:
            f.write(report_text)
        
        print(f"📝 Comprehensive report saved to {reports_dir / 'comprehensive_report.md'}")
        
        return report_text
    
    def convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {key: self.convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def get_experiment_folder(self, experiment_id: str) -> Optional[Path]:
        """Get the folder path for an experiment."""
        for exp in self.experiment_log["experiments"]:
            if str(exp["experiment_id"]).zfill(4) == experiment_id:
                return Path(exp["folder_path"])
        return None
    
    def get_experiment_metadata(self, experiment_id: str) -> Optional[Dict]:
        """Get metadata for an experiment."""
        for exp in self.experiment_log["experiments"]:
            if str(exp["experiment_id"]).zfill(4) == experiment_id:
                return exp
        return None
    
    def update_experiment_status(self, experiment_id: str, status: str, results: Dict = None):
        """Update experiment status and results."""
        for exp in self.experiment_log["experiments"]:
            if str(exp["experiment_id"]).zfill(4) == experiment_id:
                exp["status"] = status
                exp["completed_at"] = datetime.now().isoformat()
                if results:
                    exp["results"] = self.convert_to_json_serializable(results)
                break
        
        self.save_experiment_log()
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments as a DataFrame."""
        if not self.experiment_log["experiments"]:
            return pd.DataFrame()
        
        experiments_data = []
        for exp in self.experiment_log["experiments"]:
            experiments_data.append({
                "ID": f"{exp['experiment_id']:04d}",
                "Name": exp["experiment_name"],
                "Description": exp.get("description", "")[:50] + "..." if len(exp.get("description", "")) > 50 else exp.get("description", ""),
                "Status": exp.get("status", "unknown"),
                "Created": exp["created_at"][:19],  # Remove microseconds
                "Tags": ", ".join(exp.get("tags", []))
            })
        
        return pd.DataFrame(experiments_data)


# Convenience functions for easy usage
def run_benchmark_test(experiment_name: str = "benchmark_test", 
                      description: str = "Benchmark test with synthetic sinusoidal data",
                      dataset_config: Dict = None) -> str:
    """
    Quick function to run a complete benchmark test experiment.
    
    Args:
        experiment_name: Name for the experiment
        description: Description of the experiment
        dataset_config: Configuration for synthetic dataset generation
        
    Returns:
        Experiment ID
    """
    manager = ExperimentManager()
    
    # Create experiment
    exp_id = manager.create_experiment(
        experiment_name=experiment_name,
        description=description,
        tags=["benchmark", "synthetic", "sinusoidal"]
    )
    
    # Run benchmark experiment
    results = manager.run_benchmark_experiment(
        experiment_id=exp_id,
        dataset_config=dataset_config
    )
    
    return exp_id


def run_real_data_experiment(csv_file_path: str,
                           experiment_name: str = "real_data_test",
                           description: str = "Experiment on real CSV dataset",
                           feature_columns: List[str] = None,
                           target_features: int = 5,
                           normalize: bool = True) -> str:
    """
    Run a complete experiment on real CSV data (like lucy2.csv).
    
    Args:
        csv_file_path: Path to the CSV file containing real data
        experiment_name: Name for the experiment
        description: Description of the experiment
        feature_columns: List of column names to use as features (if None, auto-select numeric columns)
        target_features: Number of features to select if feature_columns is None
        normalize: Whether to normalize the data
        
    Returns:
        Experiment ID
    """
    manager = ExperimentManager()
    
    # Create experiment
    exp_id = manager.create_experiment(
        experiment_name=experiment_name,
        description=description,
        tags=["real_data", "csv", "evaluation"]
    )
    
    # Run real data experiment
    results = manager.run_real_data_experiment(
        experiment_id=exp_id,
        csv_file_path=csv_file_path,
        feature_columns=feature_columns,
        target_features=target_features,
        normalize=normalize
    )
    
    return exp_id


if __name__ == "__main__":
    # Run a test experiment
    print("🚀 Running benchmark test experiment...")
    
    exp_id = run_benchmark_test(
        experiment_name="sin_data_test",
        description="Test experiment with 5-feature sinusoidal benchmark data",
        dataset_config={
            "length": 2000,
            "window_size": 50,
            "normalize": True,
            "add_regime_changes": False
        }
    )
    
    # List all experiments
    manager = ExperimentManager()
    experiments_df = manager.list_experiments()
    print("\n📊 All Experiments:")
    print(experiments_df) 