import pennylane as qml
import torch
from torch import nn

from src.circuits.refd_QC import replicated_quantum_circuit

class ReplicatedQuantumGenerator(nn.Module):
    """
    Quantum Generator model based on the reference architecture (QC_ref.py).
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
        
        # Create the QNode using the replicated circuit
        self.qnode = qml.QNode(
            replicated_quantum_circuit, 
            q_device, 
            interface="torch", 
            diff_method="parameter-shift"
        )
        
        # Calculate and initialize the trainable parameters for the circuit
        # (2 rotation + 1 re-upload) per layer + 2 final rotations per qubit
        num_params = (3 * n_layers + 2) * n_qubits
        self.params = nn.Parameter(torch.rand(num_params) * 2 * torch.pi - torch.pi)

        # Linear layer to map quantum output to desired window size.
        # The circuit returns 2 values (X, Z) per qubit.
        self.output_layer = nn.Sequential(
            nn.Linear(2 * n_qubits, window_size),
            nn.Tanh()
        )

    def forward(self, noise_batch):
        """
        Forward pass of the generator.
        
        Args:
            noise_batch (torch.Tensor): A batch of noise vectors. Shape (batch_size, n_qubits)
        """
        batch_size = noise_batch.shape[0]
        
        quantum_outputs = []
        for i in range(batch_size):
            # Pass a single noise vector to the qnode
            sample = self.qnode(self.params, noise_batch[i], self.n_qubits, self.n_layers)
            quantum_outputs.append(torch.hstack(sample))
        
        quantum_out_stacked = torch.stack(quantum_outputs)
        
        return self.output_layer(quantum_out_stacked) 