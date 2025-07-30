#!/usr/bin/env python3
"""
Script to run QGAN experiments on the lucy2.csv real dataset.

This script demonstrates how to use the experiment management system
with real-world time series data instead of synthetic benchmark data.
"""

from experiment_manager import run_real_data_experiment

def main():
    """Run experiment on lucy2.csv dataset."""
    
    print("🔬 Starting Real Data Experiment on lucy2.csv")
    print("=" * 50)
    
    # Run experiment on lucy2.csv
    exp_id = run_real_data_experiment(
        csv_file_path="data/raw/lucy2.csv",
        experiment_name="lucy_bioprocess_analysis",
        description="QGAN evaluation on Lucy bioprocess time series data",
        feature_columns=None,  # Auto-select top 5 most variable numeric features
        target_features=5,     # Select 5 features for analysis
        normalize=True         # Normalize the data for better analysis
    )
    
    print(f"\n🎉 Experiment completed!")
    print(f"📁 Results available in: experiments/exp_{exp_id}_lucy_bioprocess_analysis")
    print(f"📝 View the comprehensive report: experiments/exp_{exp_id}_lucy_bioprocess_analysis/reports/comprehensive_report.md")
    
    # Also show how to run with specific features
    print("\n" + "=" * 50)
    print("🔬 Running Second Experiment with Specific Features")
    print("=" * 50)
    
    # Run experiment with specific bioprocess features
    exp_id_2 = run_real_data_experiment(
        csv_file_path="data/raw/lucy2.csv",
        experiment_name="lucy_bioprocess_selected",
        description="QGAN evaluation on selected Lucy bioprocess features",
        feature_columns=['TEMP_CULTURE', 'PH', 'DO', 'OD', 'CELL'],  # Specific bioprocess features
        normalize=True
    )
    
    print(f"\n🎉 Second experiment completed!")
    print(f"📁 Results available in: experiments/exp_{exp_id_2}_lucy_bioprocess_selected")
    print(f"📝 View the comprehensive report: experiments/exp_{exp_id_2}_lucy_bioprocess_selected/reports/comprehensive_report.md")
    
    return exp_id, exp_id_2

if __name__ == "__main__":
    main() 