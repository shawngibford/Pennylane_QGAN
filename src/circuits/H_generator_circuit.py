import pennylane as qml
import numpy as np
from typing import List

class QuantumCircuit:
    """
    PennyLane-based quantum circuit implementation with Hadamard initialization.
    
    The circuit consists of:
    1. Hadamard initialization layer
    2. Multiple layers of:
       - Rotation layer (Rx, Ry gates)
       - Entangling layer (CNOT gates with all-to-all topology)
       - Re-uploading layer (Rx gates)
    3. Final rotation layer (Rx, Ry gates)
    """
    
    def __init__(self, num_qubits: int, num_layers: int, device_name: str = "default.qubit"):
        """
        Initialize the quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            num_layers: Number of rotation-entangling-reuploading layer triplets
            device_name: PennyLane device name
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.device = qml.device(device_name, wires=num_qubits)
        
        # Calculate total number of parameters
        self.num_params = self.count_params()
        
        # Create measurement observables (X and Z for each qubit)
        self.observables = []
        for qubit in range(num_qubits):
            self.observables.append(qml.PauliX(qubit))
            self.observables.append(qml.PauliZ(qubit))
    
    def count_params(self) -> int:
        """
        Count the total number of parameters in the circuit.
        
        Returns:
            Total number of parameters
        """
        # Rotation layer with Rx, Ry has 2N parameters per layer
        num_params_pqc = 2 * self.num_qubits * self.num_layers
        
        # Re-uploading layer has N parameters per layer (one Rx per qubit)
        num_params_upload = self.num_layers * self.num_qubits
        
        # Final rotation layer has 2N parameters (Rx, Ry per qubit)
        num_params_final = 2 * self.num_qubits
        
        return num_params_pqc + num_params_upload + num_params_final
    
    def encoding_layer(self, noise_params: np.ndarray):
        """
        Encoding layer that prepares the initial state using noise parameters.
        Uses Rx rotations.
        
        Args:
            noise_params: Array of noise values for each qubit
        """
        for i in range(self.num_qubits):
            qml.RX(noise_params[i], wires=i)
    
    def rotation_layer(self, params: np.ndarray, start_idx: int) -> int:
        """
        Apply rotation layer with Rx and Ry gates.
        
        Args:
            params: Parameter array
            start_idx: Starting index in the parameter array
            
        Returns:
            Updated parameter index
        """
        idx = start_idx
        for qubit in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit)
            idx += 1
            qml.RY(params[idx], wires=qubit)
            idx += 1
        return idx
    
    def entangling_layer(self):
        """
        Apply entangling layer with CNOT gates using all-to-all topology.
        """
        for qubit1 in range(self.num_qubits):
            for qubit2 in range(qubit1 + 1, self.num_qubits):
                qml.CNOT(wires=[qubit1, qubit2])
    
    def reuploading_layer(self, params: np.ndarray, start_idx: int) -> int:
        """
        Apply re-uploading layer with Rx rotations.
        
        Args:
            params: Parameter array
            start_idx: Starting index in the parameter array
            
        Returns:
            Updated parameter index
        """
        idx = start_idx
        for qubit in range(self.num_qubits):
            qml.RX(params[idx], wires=qubit)
            idx += 1
        return idx
    
    def circuit(self, params: np.ndarray, noise_params: np.ndarray):
        """
        Define the complete parameterized quantum circuit.
        
        Args:
            params: Trainable parameters for the circuit
            noise_params: Noise parameters for encoding layer
        """
        # Encoding layer with noise
        self.encoding_layer(noise_params)
        
        # Hadamard initialization
        for qubit in range(self.num_qubits):
            qml.Hadamard(wires=qubit)
        
        # Main circuit layers
        idx = 0
        for layer in range(self.num_layers):
            # Rotation layer
            idx = self.rotation_layer(params, idx)
            
            # Entangling layer
            self.entangling_layer()
            
            # Re-uploading layer
            idx = self.reuploading_layer(params, idx)
        
        # Final rotation layer
        self.rotation_layer(params, idx)
    
    def create_qnode(self, observable_idx: int = 0):
        """
        Create a QNode for a specific observable.
        
        Args:
            observable_idx: Index of the observable to measure
            
        Returns:
            PennyLane QNode
        """
        @qml.qnode(self.device, diff_method="parameter-shift")
        def qnode(params, noise_params):
            self.circuit(params, noise_params)
            return qml.expval(self.observables[observable_idx])
        
        return qnode
    
    def get_expectations(self, params: np.ndarray, noise_params: np.ndarray) -> np.ndarray:
        """
        Get expectation values for all observables.
        
        Args:
            params: Trainable parameters
            noise_params: Noise parameters for encoding
            
        Returns:
            Array of expectation values
        """
        expectations = []
        
        # Create QNode for each observable
        for i in range(len(self.observables)):
            qnode = self.create_qnode(i)
            expectations.append(qnode(params, noise_params))
        
        return np.array(expectations) 