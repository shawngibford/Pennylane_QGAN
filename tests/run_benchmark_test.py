#!/usr/bin/env python3
"""
Run Benchmark Test Script

This script generates the benchmark sinusoidal dataset, saves it as CSV in data/raw,
and runs a complete QGAN evaluation experiment with comprehensive reporting.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from experiment_manager import run_benchmark_test, ExperimentManager


def main():
    """Run the complete benchmark test."""
    print("=" * 60)
    print("🚀 QGAN Benchmark Test Experiment")
    print("=" * 60)
    print()
    
    # Ensure data/raw directory exists
    data_raw_dir = Path("data/raw")
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration for the benchmark dataset
    dataset_config = {
        "length": 2000,        # 2000 time steps
        "window_size": 50,     # 50-step windows
        "normalize": True,     # Standardize the data
        "add_regime_changes": False  # Keep it simple for first test
    }
    
    print("📊 Dataset Configuration:")
    for key, value in dataset_config.items():
        print(f"  - {key}: {value}")
    print()
    
    # Run the benchmark experiment
    try:
        exp_id = run_benchmark_test(
            experiment_name="sin_data_benchmark",
            description="Comprehensive benchmark test using 5-feature sinusoidal synthetic data",
            dataset_config=dataset_config
        )
        
        print()
        print("=" * 60)
        print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print()
        
        # Show experiment details
        manager = ExperimentManager()
        exp_folder = manager.get_experiment_folder(exp_id)
        
        print("📁 Generated Files:")
        print(f"  - Raw CSV data: {exp_folder}/data/sin_data.csv")
        print(f"  - Processed dataset: {exp_folder}/data/benchmark_dataset.npz")
        print(f"  - Synthetic QGAN output: {exp_folder}/data/synthetic_qgan_output.npy")
        print(f"  - Evaluation metrics: {exp_folder}/metrics/evaluation_results.json")
        print(f"  - All plots: {exp_folder}/plots/")
        print(f"  - Comprehensive report: {exp_folder}/reports/comprehensive_report.md")
        print()
        
        print("📊 All Experiments:")
        experiments_df = manager.list_experiments()
        print(experiments_df.to_string(index=False))
        print()
        
        print("🎯 Next Steps:")
        print("1. Check the comprehensive report for detailed analysis")
        print("2. Review the generated plots for visual insights")
        print("3. Use this experiment as a baseline for QGAN development")
        print("4. Replace the synthetic data generator with your actual QGAN model")
        print()
        
        # Also copy the sin_data.csv to data/raw as requested
        import shutil
        source_csv = exp_folder / "data" / "sin_data.csv"
        target_csv = Path("data/raw/sin_data.csv")
        
        if source_csv.exists():
            shutil.copy2(source_csv, target_csv)
            print(f"📋 Copied CSV data to: {target_csv}")
        
        return exp_id
        
    except Exception as e:
        print()
        print("❌ EXPERIMENT FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    experiment_id = main()
    
    if experiment_id:
        print(f"\n🎉 Experiment {experiment_id} completed successfully!")
        print("\nTo view results:")
        print(f"  cat experiments/exp_{experiment_id}_sin_data_benchmark/reports/comprehensive_report.md")
    else:
        print("\n💥 Experiment failed. Check the error messages above.")
        sys.exit(1) 