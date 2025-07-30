import pennylane as qml
import torch
from torch import nn
import numpy as np
from typing import List, Optional, Any
from src.circuits.H_generator_circuit import QuantumCircuit

class QuantumGenerator(nn.Module):
    """
    PennyLane-based quantum generator implementation with Hadamard initialization.
    
    This generator uses a parameterized quantum circuit (PQC) with:
    - Hadamard initialization layer
    - Multiple layers of:
       - Rotation layer (Rx, Ry gates)
       - Entangling layer (CNOT gates with all-to-all topology)
       - Re-uploading layer (Rx gates)
    - Final rotation layer (Rx, Ry gates)
    - Measurements on X and Z Pauli operators
    """
    
    def __init__(self, n_qubits: int, n_layers: int, window_size: int, q_device: Any):
        """
        Initialize the quantum generator.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of layers in the circuit
            window_size: Size of the output window
            q_device: PennyLane device
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.window_size = window_size
        
        # Initialize the quantum circuit
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers, "default.qubit")
        
        # Number of parameters for the PQC
        self.num_params = self.quantum_circuit.num_params
        
        # Initialize trainable parameters
        self.params = nn.Parameter(torch.rand(self.num_params) * 2 * torch.pi - torch.pi)
        
        # Linear layer to map quantum output to desired window size
        # The circuit returns 2 values (X and Z) per qubit
        self.output_layer = nn.Sequential(
            nn.Linear(2 * n_qubits, window_size),
            nn.Tanh()
        )

    def forward(self, noise_batch):
        """
        Forward pass through the quantum generator.
        
        Args:
            noise_batch: Batch of noise vectors. Shape (batch_size, n_qubits)
            
        Returns:
            Generated samples
        """
        batch_size = noise_batch.shape[0]
        quantum_outputs = []
        
        for i in range(batch_size):
            # Get expectations for both X and Z measurements
            expectations = self.quantum_circuit.get_expectations(
                self.params.detach().numpy(),  # Convert parameters to numpy
                noise_batch[i].detach().numpy()  # Convert noise to numpy
            )
            # Convert to torch tensor
            sample = torch.tensor(expectations, dtype=torch.float32)
            quantum_outputs.append(sample)
        
        # Stack results
        quantum_out_stacked = torch.stack(quantum_outputs)
        
        # Map quantum output to the desired window size
        return self.output_layer(quantum_out_stacked)

    def generate_noise_params(self, batch_size: int = 1) -> np.ndarray:
        """
        Generate noise parameters for the encoding layer.
        Equivalent to the encoding layer in QC_ref.py.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Noise parameters for encoding layer
        """
        return np.random.uniform(0, 2 * np.pi, size=(batch_size, self.n_qubits))
    
    def generate_samples(self, num_samples: int) -> np.ndarray:
        """
        Generate multiple samples using the quantum generator.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        noise_batch = self.generate_noise_params(num_samples)
        return self.forward(torch.tensor(noise_batch, dtype=torch.float32))
    
    def generate_time_series(self, series_length: int) -> np.ndarray:
        """
        Generate a time series by concatenating multiple windows.
        This mirrors the approach in QC_ref.py where multiple windows are generated
        and concatenated to form a longer time series.
        
        Args:
            series_length: Desired length of the time series
            
        Returns:
            Generated time series
        """
        num_windows = series_length // self.window_size
        if series_length % self.window_size != 0:
            num_windows += 1
        
        # Generate windows
        windows = self.generate_samples(num_windows)
        
        # Reshape to treat each measurement as a separate time point
        # This assumes each measurement corresponds to a time step
        if self.n_qubits >= self.window_size:
            # Use first window_size measurements from each sample
            time_series = windows[:, :self.window_size].flatten()
        else:
            # Repeat measurements to match window size
            repeated_windows = []
            for window in windows:
                # Repeat the measurements to reach window_size
                repeats = self.window_size // self.n_qubits
                remainder = self.window_size % self.n_qubits
                
                repeated_window = np.tile(window, repeats)
                if remainder > 0:
                    repeated_window = np.concatenate([repeated_window, window[:remainder]])
                
                repeated_windows.append(repeated_window)
            
            time_series = np.concatenate(repeated_windows)
        
        # Truncate to exact desired length
        return time_series[:series_length]
    
    def update_parameters(self, new_params: np.ndarray):
        """
        Update the trainable parameters of the quantum circuit.
        
        Args:
            new_params: New parameter values
        """
        if len(new_params) != self.num_params:
            raise ValueError(f"Expected {self.num_params} parameters, got {len(new_params)}")
        
        self.params.data = torch.tensor(new_params, dtype=torch.float32)
    
    def get_parameters(self) -> np.ndarray:
        """
        Get the current trainable parameters.
        
        Returns:
            Current parameter values
        """
        return self.params.detach().numpy()
    
    def save_parameters(self, filepath: str):
        """
        Save the current parameters to a file.
        
        Args:
            filepath: Path to save the parameters
        """
        np.save(filepath, self.get_parameters())
    
    def load_parameters(self, filepath: str):
        """
        Load parameters from a file.
        
        Args:
            filepath: Path to load the parameters from
        """
        self.params.data = torch.tensor(np.load(filepath), dtype=torch.float32)
    
    def get_circuit_info(self) -> dict:
        """
        Get information about the quantum circuit structure.
        
        Returns:
            Dictionary containing circuit information
        """
        return {
            "num_qubits": self.n_qubits,
            "num_layers": self.n_layers,
            "num_params": self.num_params,
            "window_size": self.window_size
        }
