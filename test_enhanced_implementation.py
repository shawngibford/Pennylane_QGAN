#!/usr/bin/env python3
"""
Test Enhanced Quantum GAN Implementation
========================================

This script tests all the enhanced features of the quantum GAN implementation:
- Self-iterating discriminators
- QWGAN functionality
- Circuit benchmarking
- Feature analysis
- Comprehensive results generation
"""

import unittest
import numpy as np
import tempfile
import os
import json
from typing import Dict

# Import our modules
from enhanced_qgan import (
    SelfIteratingDiscriminator, 
    QuantumWGAN, 
    CircuitBenchmarker, 
    EnhancedQuantumGAN
)

class TestSelfIteratingDiscriminator(unittest.TestCase):
    """Test self-iterating discriminator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.data_dim = 5
        self.discriminator = SelfIteratingDiscriminator(self.data_dim)
    
    def test_initialization(self):
        """Test discriminator initialization."""
        self.assertEqual(self.discriminator.data_dim, self.data_dim)
        self.assertEqual(len(self.discriminator.performance_history), 0)
        self.assertEqual(self.discriminator.adaptation_count, 0)
        self.assertIsNotNone(self.discriminator.model)
        self.assertIsNotNone(self.discriminator.optimizer)
    
    def test_latent_representation(self):
        """Test latent representation extraction."""
        test_data = np.random.randn(10, self.data_dim)
        latent = self.discriminator.get_latent_representation(test_data)
        
        self.assertEqual(latent.shape, (10, 32))  # 32 is the latent dimension
        self.assertTrue(np.all(np.isfinite(latent)))
    
    def test_self_iteration(self):
        """Test self-iteration functionality."""
        # Test with improving performance (should not trigger adaptation)
        result1 = self.discriminator.self_iterate(0.5)
        result2 = self.discriminator.self_iterate(0.8)
        
        self.assertFalse(result1)
        self.assertFalse(result2)
        self.assertEqual(len(self.discriminator.performance_history), 2)
        
        # Test with degrading performance (should trigger adaptation)
        result3 = self.discriminator.self_iterate(0.6)  # Worse than 0.8
        
        self.assertTrue(result3)
        self.assertEqual(self.discriminator.adaptation_count, 1)
    
    def test_architecture_info(self):
        """Test architecture information retrieval."""
        info = self.discriminator._get_architecture_info()
        
        self.assertIn('layer_count', info)
        self.assertIn('total_params', info)
        self.assertIn('adaptation_count', info)
        self.assertIsInstance(info['layer_count'], int)
        self.assertIsInstance(info['total_params'], int)

class TestQuantumWGAN(unittest.TestCase):
    """Test Quantum Wasserstein Generative Associative Network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.historical_data = np.random.lognormal(0, 1, (100, 5))
        self.qwgan = QuantumWGAN(self.historical_data)
    
    def test_initialization(self):
        """Test QWGAN initialization."""
        self.assertIsNotNone(self.qwgan.historical_data)
        self.assertTrue(self.qwgan.latent_feedback)
        self.assertEqual(len(self.qwgan.feedback_history), 0)
        self.assertIsNotNone(self.qwgan.noise_bins)
        self.assertIsNotNone(self.qwgan.noise_hist)
    
    def test_histogram_noise_generation(self):
        """Test histogram-based noise generation."""
        size = (10, 5)
        noise = self.qwgan._generate_histogram_noise(size)
        
        self.assertEqual(noise.shape, size)
        self.assertTrue(np.all(np.isfinite(noise)))
        self.assertEqual(noise.dtype, np.float32)
    
    def test_latent_feedback_calculation(self):
        """Test latent feedback calculation."""
        real_latent = np.random.randn(10, 32)
        fake_latent = np.random.randn(10, 32)
        
        feedback = self.qwgan._calculate_latent_feedback(real_latent, fake_latent)
        
        self.assertIn('wasserstein_distance', feedback)
        self.assertIn('mean_distance', feedback)
        self.assertIn('std_distance', feedback)
        self.assertIn('correlation', feedback)
        
        self.assertIsInstance(feedback['wasserstein_distance'], float)
        self.assertIsInstance(feedback['mean_distance'], float)
        self.assertIsInstance(feedback['std_distance'], float)
        self.assertIsInstance(feedback['correlation'], float)
    
    def test_gradient_penalty(self):
        """Test gradient penalty calculation."""
        # Create a simple discriminator model for testing
        import tensorflow as tf
        
        discriminator_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation=None)
        ])
        
        real_batch = tf.random.normal((8, 5))
        fake_batch = tf.random.normal((8, 5))
        
        penalty = self.qwgan._gradient_penalty(discriminator_model, real_batch, fake_batch)
        
        self.assertIsInstance(penalty, tf.Tensor)
        self.assertTrue(tf.math.is_finite(penalty))

class TestCircuitBenchmarker(unittest.TestCase):
    """Test circuit benchmarking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmarker = CircuitBenchmarker()
        
        # Create mock circuits and datasets
        self.mock_circuits = {
            'VUCCA_2q_low': {
                'n_qubits': 2,
                'n_params': 8,
                'circuit_type': 'VUCCA',
                'complexity': 'low'
            },
            'UCCA_3q_medium': {
                'n_qubits': 3,
                'n_params': 15,
                'circuit_type': 'UCCA',
                'complexity': 'medium'
            }
        }
        
        self.mock_datasets = {
            'test_dataset_1': np.random.randn(100, 5),
            'test_dataset_2': np.random.randn(100, 3)
        }
    
    def test_initialization(self):
        """Test benchmarker initialization."""
        self.assertEqual(len(self.benchmarker.benchmark_results), 0)
    
    def test_complexity_limits_testing(self):
        """Test circuit complexity limits testing."""
        circuit_info = self.mock_circuits['VUCCA_2q_low']
        dataset = self.mock_datasets['test_dataset_1']
        
        complexity_limits = self.benchmarker._test_complexity_limits(circuit_info, dataset)
        
        self.assertIn('parameter_density', complexity_limits)
        self.assertIn('theoretical_max_qubits', complexity_limits)
        self.assertIn('scalability_score', complexity_limits)
        self.assertIn('complexity_efficiency', complexity_limits)
        
        # Check specific values
        self.assertEqual(complexity_limits['parameter_density'], 4.0)  # 8/2
        self.assertIsInstance(complexity_limits['theoretical_max_qubits'], int)
        self.assertGreaterEqual(complexity_limits['scalability_score'], 0.0)
        self.assertLessEqual(complexity_limits['scalability_score'], 1.0)
    
    def test_feature_capacity_testing(self):
        """Test feature learning capacity testing."""
        circuit_info = self.mock_circuits['UCCA_3q_medium']
        dataset = self.mock_datasets['test_dataset_1']
        
        feature_capacity = self.benchmarker._test_feature_capacity(circuit_info, dataset)
        
        self.assertIn('feature_qubit_ratio', feature_capacity)
        self.assertIn('encoding_efficiency', feature_capacity)
        self.assertIn('feature_compression_ratio', feature_capacity)
        self.assertIn('optimal_feature_count', feature_capacity)
        
        # Check specific values
        self.assertEqual(feature_capacity['feature_qubit_ratio'], 5/3)  # 5 features / 3 qubits
        self.assertEqual(feature_capacity['optimal_feature_count'], 5)  # min(5, 2^3)
    
    def test_entanglement_efficiency_testing(self):
        """Test entanglement efficiency testing."""
        circuit_info = self.mock_circuits['UCCA_3q_medium']
        
        entanglement_efficiency = self.benchmarker._test_entanglement_efficiency(circuit_info)
        
        self.assertIn('entanglement_density', entanglement_efficiency)
        self.assertIn('max_entanglement', entanglement_efficiency)
        self.assertIn('entanglement_efficiency', entanglement_efficiency)
        self.assertIn('entanglement_qubit_ratio', entanglement_efficiency)
        
        # Check specific values for 3 qubits
        self.assertEqual(entanglement_efficiency['max_entanglement'], 3)  # 3*(3-1)/2
        self.assertEqual(entanglement_efficiency['entanglement_qubit_ratio'], 5.0)  # 15/3
    
    def test_benchmark_circuits(self):
        """Test full circuit benchmarking."""
        results = self.benchmarker.benchmark_circuits(self.mock_circuits, self.mock_datasets)
        
        self.assertIn('VUCCA_2q_low', results)
        self.assertIn('UCCA_3q_medium', results)
        
        # Check structure of results
        for circuit_name, circuit_results in results.items():
            self.assertIn('test_dataset_1', circuit_results)
            self.assertIn('test_dataset_2', circuit_results)
            
            for dataset_name, metrics in circuit_results.items():
                self.assertIn('complexity_limits', metrics)
                self.assertIn('feature_capacity', metrics)
                self.assertIn('entanglement_efficiency', metrics)

class TestEnhancedQuantumGAN(unittest.TestCase):
    """Test enhanced quantum GAN functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.datasets = {
            'gaussian_log_A': np.random.lognormal(0, 1, (100, 5)),
            'pareto_A': np.random.pareto(2.0, (100, 5))
        }
        
        self.historical_data = np.random.lognormal(0, 1, (200, 5))
        
        self.enhanced_qgan = EnhancedQuantumGAN(
            circuit_types=['VUCCA', 'UCCA'],
            qubit_ranges=range(2, 4),
            complexity_levels=['low', 'medium'],
            datasets=self.datasets,
            historical_data=self.historical_data,
            enable_self_iteration=True,
            enable_qwgan=True,
            enable_benchmarking=True
        )
    
    def test_initialization(self):
        """Test enhanced QGAN initialization."""
        self.assertEqual(self.enhanced_qgan.circuit_types, ['VUCCA', 'UCCA'])
        self.assertEqual(list(self.enhanced_qgan.qubit_ranges), [2, 3])
        self.assertEqual(self.enhanced_qgan.complexity_levels, ['low', 'medium'])
        self.assertEqual(len(self.enhanced_qgan.datasets), 2)
        self.assertIsNotNone(self.enhanced_qgan.historical_data)
        
        # Check feature flags
        self.assertTrue(self.enhanced_qgan.enable_self_iteration)
        self.assertTrue(self.enhanced_qgan.enable_qwgan)
        self.assertTrue(self.enhanced_qgan.enable_benchmarking)
        
        # Check components
        self.assertIsNotNone(self.enhanced_qgan.benchmarker)
        self.assertIsNotNone(self.enhanced_qgan.qwgan)
    
    def test_feature_analysis(self):
        """Test feature analysis functionality."""
        analysis = self.enhanced_qgan._analyze_all_features()
        
        self.assertIn('gaussian_log_A', analysis)
        self.assertIn('pareto_A', analysis)
        
        for dataset_name, dataset_analysis in analysis.items():
            self.assertIn('correlation_matrix', dataset_analysis)
            self.assertIn('feature_statistics', dataset_analysis)
            self.assertIn('distribution_analysis', dataset_analysis)
            self.assertIn('feature_importance', dataset_analysis)
            self.assertIn('multimodality_tests', dataset_analysis)
            
            # Check correlation matrix
            corr_matrix = dataset_analysis['correlation_matrix']
            self.assertEqual(corr_matrix.shape, (5, 5))  # 5 features
            
            # Check feature statistics
            feature_stats = dataset_analysis['feature_statistics']
            self.assertEqual(len(feature_stats), 5)  # 5 features
            
            # Check distribution analysis
            dist_analysis = dataset_analysis['distribution_analysis']
            self.assertEqual(len(dist_analysis), 5)  # 5 features
    
    def test_distribution_classification(self):
        """Test distribution classification."""
        # Test normal distribution
        normal_data = np.random.normal(0, 1, 1000)
        dist_type = self.enhanced_qgan._classify_distribution(normal_data)
        self.assertIn(dist_type, ['normal', 'other'])
        
        # Test right-skewed distribution
        right_skewed = np.random.exponential(1, 1000)
        dist_type = self.enhanced_qgan._classify_distribution(right_skewed)
        self.assertIn(dist_type, ['right_skewed', 'other'])
        
        # Test left-skewed distribution
        left_skewed = -np.random.exponential(1, 1000)
        dist_type = self.enhanced_qgan._classify_distribution(left_skewed)
        self.assertIn(dist_type, ['left_skewed', 'other'])
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        dataset = np.random.randn(100, 5)
        importance = self.enhanced_qgan._calculate_feature_importance(dataset)
        
        self.assertEqual(len(importance), 5)  # 5 features
        
        for feature_name, feature_importance in importance.items():
            self.assertIn('variance_explained', feature_importance)
            self.assertIn('average_correlation', feature_importance)
            self.assertIn('importance_score', feature_importance)
            
            # Check that importance score is positive
            self.assertGreaterEqual(feature_importance['importance_score'], 0.0)
    
    def test_multimodality_testing(self):
        """Test multimodality testing."""
        # Test unimodal data
        unimodal_data = np.random.normal(0, 1, (100, 5))
        multimodality = self.enhanced_qgan._test_multimodality(unimodal_data)
        
        self.assertEqual(len(multimodality), 5)  # 5 features
        
        for feature_name, feature_multimodality in multimodality.items():
            self.assertIn('n_peaks', feature_multimodality)
            self.assertIn('is_multimodal', feature_multimodality)
            self.assertIn('peak_heights', feature_multimodality)
            
            # Unimodal data should have 1 peak
            self.assertEqual(feature_multimodality['n_peaks'], 1)
            self.assertFalse(feature_multimodality['is_multimodal'])
    
    def test_comprehensive_experiment(self):
        """Test comprehensive experiment execution."""
        results = self.enhanced_qgan.run_comprehensive_experiment()
        
        # Check that all expected results are present
        self.assertIn('benchmark_results', results)
        self.assertIn('feature_analysis', results)
        self.assertIn('qwgan_results', results)
        self.assertIn('comprehensive_table', results)
        
        # Check benchmark results
        benchmark_results = results['benchmark_results']
        self.assertGreater(len(benchmark_results), 0)
        
        # Check feature analysis
        feature_analysis = results['feature_analysis']
        self.assertEqual(len(feature_analysis), 2)  # 2 datasets
        
        # Check QWGAN results
        qwgan_results = results['qwgan_results']
        self.assertIn('histogram_noise_created', qwgan_results)
        self.assertIn('latent_feedback_enabled', qwgan_results)
        self.assertIn('training_iterations', qwgan_results)
        self.assertIn('convergence_achieved', qwgan_results)
        
        # Check comprehensive table
        comprehensive_table = results['comprehensive_table']
        self.assertIn('circuit_metrics', comprehensive_table)
        self.assertIn('feature_metrics', comprehensive_table)
        self.assertIn('performance_metrics', comprehensive_table)
        self.assertIn('distribution_metrics', comprehensive_table)
        self.assertIn('statistical_metrics', comprehensive_table)
        self.assertIn('training_metrics', comprehensive_table)

class TestIntegration(unittest.TestCase):
    """Test integration of all components."""
    
    def test_full_integration(self):
        """Test full integration of all enhanced features."""
        # Create test data
        datasets = {
            'test_gaussian': np.random.lognormal(0, 1, (50, 3)),
            'test_pareto': np.random.pareto(2.0, (50, 3))
        }
        
        historical_data = np.random.lognormal(0, 1, (100, 3))
        
        # Initialize enhanced QGAN
        enhanced_qgan = EnhancedQuantumGAN(
            circuit_types=['VUCCA'],
            qubit_ranges=range(2, 3),
            complexity_levels=['low'],
            datasets=datasets,
            historical_data=historical_data,
            enable_self_iteration=True,
            enable_qwgan=True,
            enable_benchmarking=True
        )
        
        # Run experiment
        results = enhanced_qgan.run_comprehensive_experiment()
        
        # Verify all components worked together
        self.assertIsNotNone(results)
        self.assertIn('benchmark_results', results)
        self.assertIn('feature_analysis', results)
        self.assertIn('qwgan_results', results)
        self.assertIn('comprehensive_table', results)
        
        # Check that results are consistent
        self.assertGreater(len(results['benchmark_results']), 0)
        self.assertEqual(len(results['feature_analysis']), 2)
        self.assertTrue(results['qwgan_results']['histogram_noise_created'])
        self.assertTrue(results['qwgan_results']['latent_feedback_enabled'])

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSelfIteratingDiscriminator))
    test_suite.addTest(unittest.makeSuite(TestQuantumWGAN))
    test_suite.addTest(unittest.makeSuite(TestCircuitBenchmarker))
    test_suite.addTest(unittest.makeSuite(TestEnhancedQuantumGAN))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Testing Enhanced Quantum GAN Implementation")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        print("Enhanced implementation is working correctly.")
    else:
        print("\n❌ Some tests failed!")
        print("Please check the implementation.")
    
    print("\n" + "=" * 50) 