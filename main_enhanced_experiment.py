#!/usr/bin/env python3
"""
Enhanced Quantum GAN Experiment Runner
======================================

This script runs the enhanced quantum GAN experiment with all advanced features:
- Multi-circuit benchmarking
- Feature analysis
- QWGAN training
- Self-iterating discriminators
- Comprehensive results generation
"""

import argparse
import numpy as np
import logging
from datetime import datetime
import json
import os
from typing import Dict, List

# Import our modules
from src.utils.dataset_generator import DatasetGenerator
from enhanced_qgan import EnhancedQuantumGAN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Quantum GAN Experiment')
    
    parser.add_argument('--full-features', action='store_true',
                       help='Enable all enhanced features')
    parser.add_argument('--circuit-types', nargs='+', 
                       default=['VUCCA', 'UCCA', 'IQP', 'Randomized'],
                       help='Circuit types to test')
    parser.add_argument('--qubit-range', nargs=2, type=int, default=[2, 6],
                       help='Range of qubit counts (min max)')
    parser.add_argument('--complexity-levels', nargs='+',
                       default=['low', 'medium', 'high'],
                       help='Complexity levels to test')
    parser.add_argument('--dataset-size', type=int, default=1000,
                       help='Size of generated datasets')
    parser.add_argument('--n-features', type=int, default=5,
                       help='Number of features in datasets')
    parser.add_argument('--output-dir', type=str, default='enhanced_results',
                       help='Output directory for results')
    parser.add_argument('--enable-self-iteration', action='store_true', default=True,
                       help='Enable self-iterating discriminator')
    parser.add_argument('--enable-qwgan', action='store_true', default=True,
                       help='Enable QWGAN features')
    parser.add_argument('--enable-benchmarking', action='store_true', default=True,
                       help='Enable circuit benchmarking')
    
    return parser.parse_args()

def generate_enhanced_datasets(dataset_size: int, n_features: int) -> Dict:
    """
    Generate enhanced datasets for the experiment.
    
    Args:
        dataset_size: Size of each dataset
        n_features: Number of features per dataset
        
    Returns:
        Dict: Dictionary of datasets
    """
    logger.info("Generating enhanced datasets...")
    
    # Initialize dataset generator
    dg = DatasetGenerator()
    
    datasets = {}
    
    # Generate 9 datasets as specified in ideas.txt
    # 3 Gaussian log-distributions
    datasets['gaussian_log_A'] = np.random.lognormal(0, 1, (dataset_size, n_features))
    datasets['gaussian_log_B'] = np.random.lognormal(0.5, 0.8, (dataset_size, n_features))
    datasets['gaussian_log_C'] = np.random.lognormal(-0.3, 1.2, (dataset_size, n_features))
    
    # 3 non-log distributions (very different from log)
    datasets['pareto_A'] = np.random.pareto(2.0, (dataset_size, n_features))
    datasets['exponential_A'] = np.random.exponential(1.0, (dataset_size, n_features))
    datasets['gamma_A'] = np.random.gamma(2.0, 1.0, (dataset_size, n_features))
    
    # 3 multi-modal log distributions
    # Create bimodal log-normal
    bimodal_log = np.concatenate([
        np.random.lognormal(0, 0.5, (dataset_size // 2, n_features)),
        np.random.lognormal(2, 0.5, (dataset_size // 2, n_features))
    ])
    datasets['multimodal_log_A'] = bimodal_log
    
    # Create trimodal log-normal
    trimodal_log = np.concatenate([
        np.random.lognormal(-1, 0.3, (dataset_size // 3, n_features)),
        np.random.lognormal(1, 0.3, (dataset_size // 3, n_features)),
        np.random.lognormal(3, 0.3, (dataset_size // 3, n_features))
    ])
    datasets['multimodal_log_B'] = trimodal_log
    
    # Create mixed distribution
    mixed_log = np.concatenate([
        np.random.lognormal(0, 1, (dataset_size // 2, n_features)),
        np.random.pareto(1.5, (dataset_size // 2, n_features))
    ])
    datasets['multimodal_log_C'] = mixed_log
    
    logger.info(f"Generated {len(datasets)} datasets with {n_features} features each")
    
    return datasets

def create_historical_data(dataset_size: int, n_features: int) -> np.ndarray:
    """
    Create historical data for QWGAN.
    
    Args:
        dataset_size: Size of historical data
        n_features: Number of features
        
    Returns:
        np.ndarray: Historical data
    """
    logger.info("Creating historical data for QWGAN...")
    
    # Create realistic historical data with multiple distributions
    historical_data = np.concatenate([
        np.random.lognormal(0, 1, (dataset_size // 3, n_features)),
        np.random.pareto(2.0, (dataset_size // 3, n_features)),
        np.random.exponential(1.0, (dataset_size // 3, n_features))
    ])
    
    logger.info(f"Created historical data with shape {historical_data.shape}")
    return historical_data

def save_results(results: Dict, output_dir: str):
    """
    Save experiment results to files.
    
    Args:
        results: Experiment results
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive results
    results_file = os.path.join(output_dir, f'enhanced_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x))
        json.dump(json_results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(output_dir, f'enhanced_summary_{timestamp}.txt')
    with open(summary_file, 'w') as f:
        f.write("Enhanced Quantum GAN Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Results saved to: {results_file}\n\n")
        
        # Write summary of results
        if 'benchmark_results' in results:
            f.write(f"Circuit Benchmarking: {len(results['benchmark_results'])} circuits tested\n")
        
        if 'feature_analysis' in results:
            f.write(f"Feature Analysis: {len(results['feature_analysis'])} datasets analyzed\n")
        
        if 'qwgan_results' in results:
            f.write("QWGAN Training: Completed\n")
        
        if 'comprehensive_table' in results:
            f.write("Comprehensive Table: Generated\n")
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Results file: {results_file}")
    logger.info(f"Summary file: {summary_file}")

def print_comprehensive_table(results: Dict):
    """
    Print comprehensive results table.
    
    Args:
        results: Experiment results
    """
    print("\n" + "="*80)
    print("ENHANCED QUANTUM GAN EXPERIMENT RESULTS")
    print("="*80)
    
    if 'benchmark_results' in results:
        print(f"\n📊 CIRCUIT BENCHMARKING RESULTS")
        print("-" * 40)
        for circuit_name, circuit_results in results['benchmark_results'].items():
            print(f"\n🔧 {circuit_name}:")
            for dataset_name, metrics in circuit_results.items():
                print(f"  📈 {dataset_name}:")
                if 'complexity_limits' in metrics:
                    complexity = metrics['complexity_limits']
                    print(f"    - Parameter Density: {complexity.get('parameter_density', 'N/A'):.3f}")
                    print(f"    - Scalability Score: {complexity.get('scalability_score', 'N/A'):.3f}")
                
                if 'feature_capacity' in metrics:
                    capacity = metrics['feature_capacity']
                    print(f"    - Feature/Qubit Ratio: {capacity.get('feature_qubit_ratio', 'N/A'):.3f}")
                    print(f"    - Encoding Efficiency: {capacity.get('encoding_efficiency', 'N/A'):.3f}")
    
    if 'feature_analysis' in results:
        print(f"\n📈 FEATURE ANALYSIS RESULTS")
        print("-" * 40)
        for dataset_name, analysis in results['feature_analysis'].items():
            print(f"\n📊 {dataset_name}:")
            
            # Distribution analysis
            if 'distribution_analysis' in analysis:
                dist_analysis = analysis['distribution_analysis']
                normal_features = sum(1 for f in dist_analysis.values() 
                                    if f.get('distribution_type') == 'normal')
                print(f"  - Normal Features: {normal_features}/{len(dist_analysis)}")
            
            # Feature importance
            if 'feature_importance' in analysis:
                importance = analysis['feature_importance']
                avg_importance = np.mean([f['importance_score'] for f in importance.values()])
                print(f"  - Average Feature Importance: {avg_importance:.3f}")
            
            # Multimodality
            if 'multimodality_tests' in analysis:
                multimodality = analysis['multimodality_tests']
                multimodal_features = sum(1 for f in multimodality.values() 
                                        if f.get('is_multimodal', False))
                print(f"  - Multimodal Features: {multimodal_features}/{len(multimodality)}")
    
    if 'qwgan_results' in results:
        print(f"\n🌊 QWGAN TRAINING RESULTS")
        print("-" * 40)
        qwgan_results = results['qwgan_results']
        print(f"  - Histogram Noise Created: {qwgan_results.get('histogram_noise_created', False)}")
        print(f"  - Latent Feedback Enabled: {qwgan_results.get('latent_feedback_enabled', False)}")
        print(f"  - Training Iterations: {qwgan_results.get('training_iterations', 0)}")
        print(f"  - Convergence Achieved: {qwgan_results.get('convergence_achieved', False)}")
    
    if 'comprehensive_table' in results:
        print(f"\n📋 COMPREHENSIVE METRICS TABLE")
        print("-" * 40)
        table = results['comprehensive_table']
        
        if 'circuit_metrics' in table:
            print(f"  - Circuit Metrics: {len(table['circuit_metrics'])} entries")
        
        if 'feature_metrics' in table:
            print(f"  - Feature Metrics: {len(table['feature_metrics'])} entries")
        
        if 'performance_metrics' in table:
            print(f"  - Performance Metrics: {len(table['performance_metrics'])} entries")
    
    print("\n" + "="*80)

def main():
    """Main experiment runner."""
    args = parse_arguments()
    
    logger.info("Starting Enhanced Quantum GAN Experiment")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Step 1: Generate datasets
        datasets = generate_enhanced_datasets(args.dataset_size, args.n_features)
        
        # Step 2: Create historical data for QWGAN
        historical_data = create_historical_data(args.dataset_size, args.n_features)
        
        # Step 3: Initialize enhanced QGAN
        enhanced_qgan = EnhancedQuantumGAN(
            circuit_types=args.circuit_types,
            qubit_ranges=range(args.qubit_range[0], args.qubit_range[1] + 1),
            complexity_levels=args.complexity_levels,
            datasets=datasets,
            historical_data=historical_data,
            enable_self_iteration=args.enable_self_iteration,
            enable_qwgan=args.enable_qwgan,
            enable_benchmarking=args.enable_benchmarking
        )
        
        # Step 4: Run comprehensive experiment
        logger.info("Running comprehensive enhanced experiment...")
        results = enhanced_qgan.run_comprehensive_experiment()
        
        # Step 5: Print results
        print_comprehensive_table(results)
        
        # Step 6: Save results
        save_results(results, args.output_dir)
        
        logger.info("Enhanced experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 