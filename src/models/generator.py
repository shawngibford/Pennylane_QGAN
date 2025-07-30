import pennylane as qml
import torch
from torch import nn

from src.circuits.generator_circuit import generator_circuit_fn

class QuantumGenerator(nn.Module):
    """
    Quantum Generator model that wraps the quantum circuit in a PyTorch layer.
    """
    def __init__(self, n_qubits, n_layers, window_size, q_device):
        """
        Args:
            n_qubits (int): Number of qubits.
            n_layers (int): Number of layers in the circuit.
            window_size (int): The size of the output window.
            q_device (qml.device): PennyLane device to run the circuit on.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create the QNode
        self.qnode = qml.QNode(generator_circuit_fn, q_device, interface="torch", diff_method="parameter-shift")
        
        # Initialize and register the trainable parameters for the quantum circuit
        num_params = self.n_qubits * self.n_layers * 3
        self.params = nn.Parameter(torch.rand(num_params) * 2 * torch.pi - torch.pi)

        # Linear layer to map quantum output to desired window size
        self.output_layer = nn.Sequential(
            nn.Linear(n_qubits, window_size),
            nn.Tanh()
        )

    def forward(self, noise_batch):
        """
        Forward pass of the generator.
        
        Args:
            noise_batch (torch.Tensor): A batch of noise vectors. Shape (batch_size, n_qubits)
        """
        batch_size = noise_batch.shape[0]
        
        # Process noise batch through the quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            # Pass a single noise vector to the qnode and convert output to float32 tensor
            sample = torch.tensor(self.qnode(self.params, noise_batch[i], self.n_qubits, self.n_layers), dtype=torch.float32)
            quantum_outputs.append(sample)
        
        # Stack results to form a tensor of shape (batch_size, n_qubits)
        quantum_out_stacked = torch.stack(quantum_outputs)
        
        # Map quantum output to the desired window size
        return self.output_layer(quantum_out_stacked) 