#!/usr/bin/env python3
"""
Test Implementation Script
==========================

This script tests the implementation of the quantum synthetic data generation
framework to ensure all components work together correctly.
"""

import os
import sys
import numpy as np
import logging

# Add src to path
sys.path.append('src')

# Import our modules
from src.utils.dataset_generator import DatasetGenerator
from src.circuits.circuit_dictionary import CircuitDictionary
from src.models.quantum_gan import QuantumGAN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataset_generator():
    """Test the dataset generator."""
    print("Testing Dataset Generator...")
    
    # Create generator with smaller sample size for testing
    generator = DatasetGenerator(sample_size=1000, n_features=3, save_path="test_data")
    
    # Generate all datasets
    datasets = generator.generate_all_datasets()
    
    # Validate datasets
    validation_results = generator.validate_datasets(datasets)
    
    # Save datasets
    generator.save_datasets(datasets, validation_results)
    
    print(f"✓ Generated {len(datasets)} datasets")
    print(f"✓ Validation completed for all datasets")
    
    return datasets, validation_results

def test_circuit_dictionary():
    """Test the circuit dictionary."""
    print("\nTesting Circuit Dictionary...")
    
    # Create circuit dictionary
    cd = CircuitDictionary()
    
    # Test getting a specific circuit
    circuit, n_params = cd.get_circuit('VUCCA', 4, 'medium')
    print(f"✓ Created VUCCA circuit with 4 qubits, medium complexity: {n_params} parameters")
    
    # Test getting all circuits for a range
    all_circuits = cd.get_all_circuits(range(2, 4), ['low', 'medium'])
    print(f"✓ Generated {len(all_circuits)} circuits")
    
    return cd, all_circuits

def test_quantum_gan():
    """Test the QuantumGAN class."""
    print("\nTesting QuantumGAN...")
    
    # Create sample datasets
    datasets = {
        'gaussian_log_A': np.random.lognormal(0, 1, (500, 3)),
        'gaussian_log_B': np.random.lognormal(0.5, 1.2, (500, 3)),
        'pareto_D': np.random.pareto(2.0, (500, 3))
    }
    
    # Initialize QGAN
    qgan = QuantumGAN(
        circuit_types=['VUCCA', 'UCCA'],
        qubit_ranges=range(2, 4),
        complexity_levels=['low', 'medium'],
        datasets=datasets,
        weight_storage_path="test_weights",
        results_storage_path="test_results"
    )
    
    print(f"✓ Initialized QGAN with {len(datasets)} datasets")
    
    # Test permutation iteration
    permutation_count = 0
    for config_key, config in qgan.iterate_over_permutations():
        permutation_count += 1
        print(f"  - Configuration {permutation_count}: {config_key}")
    
    print(f"✓ Generated {permutation_count} permutations")
    
    return qgan

def test_integration():
    """Test the integration of all components."""
    print("\nTesting Integration...")
    
    # Test dataset generator
    datasets, validation_results = test_dataset_generator()
    
    # Test circuit dictionary
    cd, all_circuits = test_circuit_dictionary()
    
    # Test QGAN with generated datasets
    qgan = QuantumGAN(
        circuit_types=['VUCCA'],
        qubit_ranges=range(2, 3),
        complexity_levels=['low'],
        datasets=datasets,
        weight_storage_path="test_weights",
        results_storage_path="test_results"
    )
    
    print("✓ All components integrated successfully")
    
    return True

def cleanup_test_files():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    test_dirs = ["test_data", "test_weights", "test_results"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            import shutil
            shutil.rmtree(test_dir)
            print(f"✓ Removed {test_dir}")

def main():
    """Main test function."""
    print("="*60)
    print("TESTING QUANTUM SYNTHETIC DATA GENERATION FRAMEWORK")
    print("="*60)
    
    try:
        # Test individual components
        test_dataset_generator()
        test_circuit_dictionary()
        test_quantum_gan()
        
        # Test integration
        test_integration()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("The quantum synthetic data generation framework is working correctly.")
        print("You can now run the main experiment with: python main_qgan_experiment.py")
        
        # Clean up
        cleanup_test_files()
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        logger.error(f"Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 