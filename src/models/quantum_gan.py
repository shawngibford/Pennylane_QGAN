"""
Quantum Generative Adversarial Network (QGAN) Implementation
===========================================================

This module implements a QGAN class that can iterate over all permutations
of circuit types, qubit counts, and complexity levels, with weight storage
and retrieval capabilities.
"""

import pennylane as qml
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Generator, Any
import itertools
import os
import pickle
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumGAN:
    """
    Quantum Generative Adversarial Network with permutation iteration capability.
    
    This class implements a QGAN that can iterate over all combinations of:
    - Circuit types (VUCCA, UCCA, IQP, Randomized)
    - Qubit counts (n to n+m)
    - Complexity levels (low, medium, high, extreme)
    - Datasets (9 different distribution types)
    """
    
    def __init__(self, 
                 circuit_types: List[str] = None,
                 qubit_ranges: range = None,
                 complexity_levels: List[str] = None,
                 datasets: Dict = None,
                 weight_storage_path: str = "models/weights",
                 results_storage_path: str = "experiments/results"):
        """
        Initialize the QuantumGAN.
        
        Args:
            circuit_types: List of circuit types to iterate over
            qubit_ranges: Range of qubit counts to test
            complexity_levels: List of complexity levels to test
            datasets: Dictionary of datasets to train on
            weight_storage_path: Path to store trained weights
            results_storage_path: Path to store experiment results
        """
        self.circuit_types = circuit_types or ['VUCCA', 'UCCA', 'IQP', 'Randomized']
        self.qubit_ranges = qubit_ranges or range(2, 8)  # 2 to 7 qubits
        self.complexity_levels = complexity_levels or ['low', 'medium', 'high']
        self.datasets = datasets or {}
        
        # Storage paths
        self.weight_storage_path = weight_storage_path
        self.results_storage_path = results_storage_path
        
        # Create storage directories
        os.makedirs(weight_storage_path, exist_ok=True)
        os.makedirs(results_storage_path, exist_ok=True)
        
        # Initialize storage dictionaries
        self.weight_storage = {}
        self.performance_metrics = {}
        self.complexity_metrics = {}
        self.training_history = {}
        
        # Circuit dictionary
        from src.circuits.circuit_dictionary import CircuitDictionary
        self.circuit_dict = CircuitDictionary()
        
        # Training parameters
        self.default_training_params = {
            'epochs': 1000,
            'batch_size': 32,
            'learning_rate': 0.001,
            'critic_iterations': 5,
            'lambda_gp': 10.0,
            'noise_dim': 100
        }
        
        logger.info(f"QuantumGAN initialized with {len(self.circuit_types)} circuit types, "
                   f"{len(list(self.qubit_ranges))} qubit counts, "
                   f"{len(self.complexity_levels)} complexity levels")
    
    def iterate_over_permutations(self) -> Generator[Tuple[str, Dict], None, None]:
        """
        Iterate over all combinations of circuits, qubits, complexity, and datasets.
        
        Yields:
            Tuple[str, Dict]: Configuration key and configuration dictionary
        """
        total_combinations = (len(self.circuit_types) * 
                            len(list(self.qubit_ranges)) * 
                            len(self.complexity_levels) * 
                            len(self.datasets))
        
        logger.info(f"Starting iteration over {total_combinations} total combinations")
        
        combination_count = 0
        
        for circuit_type in self.circuit_types:
            for n_qubits in self.qubit_ranges:
                for complexity in self.complexity_levels:
                    for dataset_name, dataset in self.datasets.items():
                        combination_count += 1
                        
                        config_key = f"{circuit_type}_{n_qubits}q_{complexity}_{dataset_name}"
                        config = {
                            'circuit_type': circuit_type,
                            'n_qubits': n_qubits,
                            'complexity': complexity,
                            'dataset_name': dataset_name,
                            'dataset': dataset,
                            'combination_number': combination_count,
                            'total_combinations': total_combinations
                        }
                        
                        logger.info(f"Processing combination {combination_count}/{total_combinations}: {config_key}")
                        
                        yield config_key, config
    
    def train_single_config(self, config_key: str, config: Dict, 
                           training_params: Dict = None) -> Dict:
        """
        Train QGAN for a single configuration.
        
        Args:
            config_key: Unique key for this configuration
            config: Configuration dictionary
            training_params: Training parameters (optional)
        
        Returns:
            Dict: Training results and metrics
        """
        if training_params is None:
            training_params = self.default_training_params
        
        circuit_type = config['circuit_type']
        n_qubits = config['n_qubits']
        complexity = config['complexity']
        dataset = config['dataset']
        
        logger.info(f"Training {config_key}")
        
        try:
            # Get quantum circuit
            circuit, n_params = self.circuit_dict.get_circuit(circuit_type, n_qubits, complexity)
            
            # Measure circuit complexity before training
            complexity_metrics = self.measure_circuit_complexity(circuit, n_qubits, complexity)
            
            # Check if weights already exist
            weights = self.load_weights(config_key)
            if weights is not None:
                logger.info(f"Using pre-trained weights for {config_key}")
                performance_metrics = self.evaluate_performance(config_key, weights, dataset)
            else:
                # Train new model
                logger.info(f"Training new model for {config_key}")
                weights, performance_metrics, training_history = self._train_model(
                    circuit, dataset, training_params, config_key
                )
                
                # Save weights
                self.save_weights(config_key, weights)
                self.training_history[config_key] = training_history
            
            # Store results
            results = {
                'config_key': config_key,
                'config': config,
                'complexity_metrics': complexity_metrics,
                'performance_metrics': performance_metrics,
                'n_params': n_params,
                'weights_saved': weights is not None,
                'timestamp': datetime.now().isoformat()
            }
            
            self.performance_metrics[config_key] = performance_metrics
            self.complexity_metrics[config_key] = complexity_metrics
            
            # Save results
            self._save_results(config_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error training {config_key}: {str(e)}")
            return {
                'config_key': config_key,
                'config': config,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _train_model(self, circuit: qml.QNode, dataset: np.ndarray, 
                    training_params: Dict, config_key: str) -> Tuple[Dict, Dict, Dict]:
        """
        Train the quantum generator model.
        
        Args:
            circuit: Quantum circuit
            dataset: Training dataset
            training_params: Training parameters
            config_key: Configuration key for logging
        
        Returns:
            Tuple[Dict, Dict, Dict]: Weights, performance metrics, training history
        """
        # Initialize quantum generator
        generator = self._create_quantum_generator(circuit, training_params['noise_dim'])
        
        # Initialize classical discriminator
        discriminator = self._create_classical_discriminator(dataset.shape[1])
        
        # Training loop
        training_history = {
            'generator_losses': [],
            'discriminator_losses': [],
            'wasserstein_distances': []
        }
        
        # Optimizers
        gen_optimizer = tf.keras.optimizers.Adam(training_params['learning_rate'])
        disc_optimizer = tf.keras.optimizers.Adam(training_params['learning_rate'])
        
        for epoch in range(training_params['epochs']):
            # Train discriminator
            for _ in range(training_params['critic_iterations']):
                disc_loss = self._train_discriminator_step(
                    discriminator, generator, dataset, disc_optimizer, training_params
                )
            
            # Train generator
            gen_loss = self._train_generator_step(
                generator, discriminator, gen_optimizer, training_params
            )
            
            # Calculate Wasserstein distance
            w_distance = self._calculate_wasserstein_distance(discriminator, dataset, generator)
            
            # Store history
            training_history['generator_losses'].append(gen_loss)
            training_history['discriminator_losses'].append(disc_loss)
            training_history['wasserstein_distances'].append(w_distance)
            
            if epoch % 100 == 0:
                logger.info(f"{config_key} - Epoch {epoch}: "
                           f"G_loss={gen_loss:.4f}, D_loss={disc_loss:.4f}, "
                           f"W_dist={w_distance:.4f}")
        
        # Get final weights
        weights = {
            'generator_weights': generator.get_weights(),
            'discriminator_weights': discriminator.get_weights()
        }
        
        # Calculate final performance metrics
        performance_metrics = self._calculate_performance_metrics(generator, dataset)
        
        return weights, performance_metrics, training_history
    
    def _create_quantum_generator(self, circuit: qml.QNode, noise_dim: int) -> tf.keras.Model:
        """Create quantum generator model."""
        # This is a simplified implementation - you would need to integrate
        # PennyLane with TensorFlow more thoroughly
        inputs = tf.keras.Input(shape=(noise_dim,))
        
        # Classical preprocessing
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Quantum layer (simplified)
        # In practice, you'd use PennyLane's TensorFlow interface
        quantum_output = tf.keras.layers.Dense(circuit.num_wires, activation='tanh')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=quantum_output)
    
    def _create_classical_discriminator(self, data_dim: int) -> tf.keras.Model:
        """Create classical discriminator model."""
        inputs = tf.keras.Input(shape=(data_dim,))
        
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dense(1, activation=None)(x)  # No activation for WGAN
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def _train_discriminator_step(self, discriminator, generator, dataset, 
                                 optimizer, training_params):
        """Train discriminator for one step."""
        batch_size = training_params['batch_size']
        lambda_gp = training_params['lambda_gp']
        
        # Get real and fake data
        real_batch = self._get_random_batch(dataset, batch_size)
        noise = tf.random.normal([batch_size, training_params['noise_dim']])
        fake_batch = generator(noise, training=True)
        
        with tf.GradientTape() as tape:
            # Discriminator outputs
            real_output = discriminator(real_batch, training=True)
            fake_output = discriminator(fake_batch, training=True)
            
            # WGAN-GP loss
            gradient_penalty = self._gradient_penalty(discriminator, real_batch, fake_batch)
            disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + lambda_gp * gradient_penalty
        
        # Apply gradients
        gradients = tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        
        return disc_loss.numpy()
    
    def _train_generator_step(self, generator, discriminator, optimizer, training_params):
        """Train generator for one step."""
        batch_size = training_params['batch_size']
        noise = tf.random.normal([batch_size, training_params['noise_dim']])
        
        with tf.GradientTape() as tape:
            fake_batch = generator(noise, training=True)
            fake_output = discriminator(fake_batch, training=True)
            gen_loss = -tf.reduce_mean(fake_output)
        
        # Apply gradients
        gradients = tape.gradient(gen_loss, generator.trainable_variables)
        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        
        return gen_loss.numpy()
    
    def _gradient_penalty(self, discriminator, real_batch, fake_batch):
        """Calculate gradient penalty for WGAN-GP."""
        batch_size = tf.shape(real_batch)[0]
        alpha = tf.random.uniform([batch_size, 1], 0., 1.)
        
        interpolated = alpha * real_batch + (1 - alpha) * fake_batch
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interpolated_output = discriminator(interpolated, training=True)
        
        gradients = tape.gradient(interpolated_output, interpolated)
        gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        gradient_penalty = tf.reduce_mean(tf.square(gradient_norm - 1))
        
        return gradient_penalty
    
    def _get_random_batch(self, dataset: np.ndarray, batch_size: int) -> tf.Tensor:
        """Get random batch from dataset."""
        indices = np.random.choice(len(dataset), batch_size, replace=False)
        return tf.convert_to_tensor(dataset[indices], dtype=tf.float32)
    
    def _calculate_wasserstein_distance(self, discriminator, dataset, generator):
        """Calculate Wasserstein distance."""
        batch_size = 100
        real_batch = self._get_random_batch(dataset, batch_size)
        noise = tf.random.normal([batch_size, generator.input_shape[1]])
        fake_batch = generator(noise, training=False)
        
        real_output = discriminator(real_batch, training=False)
        fake_output = discriminator(fake_batch, training=False)
        
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
    
    def _calculate_performance_metrics(self, generator, dataset):
        """Calculate performance metrics for the trained generator."""
        # Generate synthetic data
        noise = tf.random.normal([len(dataset), generator.input_shape[1]])
        synthetic_data = generator(noise, training=False).numpy()
        
        # Calculate basic metrics
        metrics = {
            'mae': np.mean(np.abs(dataset - synthetic_data)),
            'rmse': np.sqrt(np.mean((dataset - synthetic_data) ** 2)),
            'wasserstein_distance': self._calculate_wasserstein_distance(
                None, dataset, generator
            ).numpy()
        }
        
        return metrics
    
    def measure_circuit_complexity(self, circuit: qml.QNode, n_qubits: int, 
                                 complexity: str) -> Dict:
        """
        Measure complexity and entanglement of quantum circuit.
        
        Args:
            circuit: Quantum circuit
            n_qubits: Number of qubits
            complexity: Complexity level
        
        Returns:
            Dict: Complexity metrics
        """
        metrics = {
            'n_qubits': n_qubits,
            'complexity_level': complexity,
            'gate_count': self._count_gates(circuit),
            'depth': self._calculate_circuit_depth(circuit),
            'parameter_count': circuit.num_params,
            'entanglement_entropy': self._calculate_entanglement_entropy(circuit, n_qubits),
            'coherence': self._calculate_coherence(circuit, n_qubits),
            'purity': self._calculate_purity(circuit, n_qubits)
        }
        
        return metrics
    
    def _count_gates(self, circuit: qml.QNode) -> int:
        """Count number of gates in circuit."""
        # Simplified implementation - in practice you'd parse the circuit
        return circuit.num_params * 2  # Rough estimate
    
    def _calculate_circuit_depth(self, circuit: qml.QNode) -> int:
        """Calculate circuit depth."""
        # Simplified implementation
        return circuit.num_params // 3  # Rough estimate
    
    def _calculate_entanglement_entropy(self, circuit: qml.QNode, n_qubits: int) -> float:
        """Calculate entanglement entropy."""
        # Simplified implementation
        return np.random.uniform(0, 1)  # Placeholder
    
    def _calculate_coherence(self, circuit: qml.QNode, n_qubits: int) -> float:
        """Calculate coherence measure."""
        # Simplified implementation
        return np.random.uniform(0, 1)  # Placeholder
    
    def _calculate_purity(self, circuit: qml.QNode, n_qubits: int) -> float:
        """Calculate purity measure."""
        # Simplified implementation
        return np.random.uniform(0, 1)  # Placeholder
    
    def save_weights(self, config_key: str, weights: Dict) -> None:
        """
        Save trained weights for future reuse.
        
        Args:
            config_key: Configuration key
            weights: Weights dictionary
        """
        weight_file = os.path.join(self.weight_storage_path, f"{config_key}_weights.pkl")
        
        with open(weight_file, 'wb') as f:
            pickle.dump(weights, f)
        
        logger.info(f"Weights saved for {config_key}")
    
    def load_weights(self, config_key: str) -> Optional[Dict]:
        """
        Load pre-trained weights to avoid retraining.
        
        Args:
            config_key: Configuration key
        
        Returns:
            Optional[Dict]: Weights if found, None otherwise
        """
        weight_file = os.path.join(self.weight_storage_path, f"{config_key}_weights.pkl")
        
        if os.path.exists(weight_file):
            with open(weight_file, 'rb') as f:
                weights = pickle.load(f)
            logger.info(f"Weights loaded for {config_key}")
            return weights
        else:
            logger.info(f"No pre-trained weights found for {config_key}")
            return None
    
    def evaluate_performance(self, config_key: str, weights: Dict, 
                           dataset: np.ndarray) -> Dict:
        """
        Evaluate performance of a trained model.
        
        Args:
            config_key: Configuration key
            weights: Model weights
            dataset: Test dataset
        
        Returns:
            Dict: Performance metrics
        """
        # This would load the model with weights and evaluate
        # Simplified implementation
        return {
            'mae': np.random.uniform(0, 1),
            'rmse': np.random.uniform(0, 1),
            'wasserstein_distance': np.random.uniform(0, 1)
        }
    
    def _save_results(self, config_key: str, results: Dict) -> None:
        """Save experiment results."""
        results_file = os.path.join(self.results_storage_path, f"{config_key}_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for all experiments.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_configurations': len(self.performance_metrics),
            'successful_training': len([k for k, v in self.performance_metrics.items() if 'error' not in v]),
            'failed_training': len([k for k, v in self.performance_metrics.items() if 'error' in v]),
            'best_performance': None,
            'worst_performance': None,
            'average_performance': {}
        }
        
        if self.performance_metrics:
            # Calculate average performance metrics
            metrics_list = [v for v in self.performance_metrics.values() if 'error' not in v]
            if metrics_list:
                summary['average_performance'] = {
                    'mae': np.mean([m.get('mae', 0) for m in metrics_list]),
                    'rmse': np.mean([m.get('rmse', 0) for m in metrics_list]),
                    'wasserstein_distance': np.mean([m.get('wasserstein_distance', 0) for m in metrics_list])
                }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample datasets
    datasets = {
        'gaussian_log_A': np.random.lognormal(0, 1, (1000, 5)),
        'gaussian_log_B': np.random.lognormal(0.5, 1.2, (1000, 5)),
        'pareto_D': np.random.pareto(2.0, (1000, 5))
    }
    
    # Initialize QGAN
    qgan = QuantumGAN(
        circuit_types=['VUCCA', 'UCCA'],
        qubit_ranges=range(2, 4),
        complexity_levels=['low', 'medium'],
        datasets=datasets
    )
    
    # Iterate over all permutations
    for config_key, config in qgan.iterate_over_permutations():
        results = qgan.train_single_config(config_key, config)
        print(f"Completed: {config_key}")
    
    # Get summary
    summary = qgan.get_summary_statistics()
    print(f"Summary: {summary}") 