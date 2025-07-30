"""
Quantum Circuit Dictionary for QGAN Implementation
=================================================

This module implements a dictionary of quantum circuits for n to n+m qubits,
including VUCCA, UCCA, IQP, and randomized circuits with varying complexity levels.
"""

import pennylane as qml
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools

class CircuitDictionary:
    """
    Dictionary of quantum circuits for different complexity levels and qubit counts.
    """
    
    def __init__(self):
        self.complexity_levels = {
            'low': {'layers': 2, 'entanglement': 'linear', 'rotation_depth': 1},
            'medium': {'layers': 4, 'entanglement': 'circular', 'rotation_depth': 2},
            'high': {'layers': 8, 'entanglement': 'all_to_all', 'rotation_depth': 3},
            'extreme': {'layers': 16, 'entanglement': 'custom', 'rotation_depth': 4}
        }
        
        self.entanglement_patterns = {
            'linear': 'nearest_neighbor',
            'circular': 'ring',
            'all_to_all': 'complete_graph',
            'custom': 'adaptive_based_on_data'
        }
        
        self.circuit_types = ['VUCCA', 'UCCA', 'IQP', 'Randomized']
    
    def create_vucca_circuit(self, n_qubits: int, n_layers: int, entanglement: str = 'linear') -> qml.QNode:
        """
        Create Variational Unitary Coupled Cluster Ansatz circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            entanglement: Entanglement pattern ('linear', 'circular', 'all_to_all')
        
        Returns:
            qml.QNode: PennyLane quantum node
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def vucca_circuit(params):
            # Initialize in computational basis
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # VUCCA layers
            param_idx = 0
            for layer in range(n_layers):
                # Single qubit rotations
                for i in range(n_qubits):
                    qml.RX(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                # Entangling gates based on pattern
                if entanglement == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        qml.CRZ(params[param_idx], wires=[i, i+1])
                        param_idx += 1
                
                elif entanglement == 'circular':
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i+1) % n_qubits])
                        qml.CRZ(params[param_idx], wires=[i, (i+1) % n_qubits])
                        param_idx += 1
                
                elif entanglement == 'all_to_all':
                    for i, j in itertools.combinations(range(n_qubits), 2):
                        qml.CNOT(wires=[i, j])
                        qml.CRZ(params[param_idx], wires=[i, j])
                        param_idx += 1
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return vucca_circuit
    
    def create_ucca_circuit(self, n_qubits: int, n_layers: int, entanglement: str = 'linear') -> qml.QNode:
        """
        Create Unitary Coupled Cluster Ansatz circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            entanglement: Entanglement pattern
        
        Returns:
            qml.QNode: PennyLane quantum node
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def ucca_circuit(params):
            # Initialize in computational basis
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # UCCA layers with more sophisticated entangling structure
            param_idx = 0
            for layer in range(n_layers):
                # Single qubit rotations
                for i in range(n_qubits):
                    qml.RX(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                
                # UCCA-specific entangling structure
                if entanglement == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        qml.CRZ(params[param_idx], wires=[i, i+1])
                        param_idx += 1
                        qml.CRY(params[param_idx], wires=[i, i+1])
                        param_idx += 1
                
                elif entanglement == 'circular':
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i+1) % n_qubits])
                        qml.CRZ(params[param_idx], wires=[i, (i+1) % n_qubits])
                        param_idx += 1
                        qml.CRY(params[param_idx], wires=[i, (i+1) % n_qubits])
                        param_idx += 1
                
                elif entanglement == 'all_to_all':
                    for i, j in itertools.combinations(range(n_qubits), 2):
                        qml.CNOT(wires=[i, j])
                        qml.CRZ(params[param_idx], wires=[i, j])
                        param_idx += 1
                        qml.CRY(params[param_idx], wires=[i, j])
                        param_idx += 1
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return ucca_circuit
    
    def create_iqp_circuit(self, n_qubits: int, n_layers: int, entanglement: str = 'linear') -> qml.QNode:
        """
        Create Instantaneous Quantum Polynomial circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            entanglement: Entanglement pattern
        
        Returns:
            qml.QNode: PennyLane quantum node
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def iqp_circuit(params):
            # IQP initialization - Hadamard on all qubits
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # IQP layers with diagonal entangling gates
            param_idx = 0
            for layer in range(n_layers):
                # Diagonal entangling gates (IQP characteristic)
                if entanglement == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CRZ(params[param_idx], wires=[i, i+1])
                        param_idx += 1
                
                elif entanglement == 'circular':
                    for i in range(n_qubits):
                        qml.CRZ(params[param_idx], wires=[i, (i+1) % n_qubits])
                        param_idx += 1
                
                elif entanglement == 'all_to_all':
                    for i, j in itertools.combinations(range(n_qubits), 2):
                        qml.CRZ(params[param_idx], wires=[i, j])
                        param_idx += 1
                
                # Single qubit rotations
                for i in range(n_qubits):
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
            
            # Final Hadamard layer (IQP characteristic)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return iqp_circuit
    
    def create_randomized_circuit(self, n_qubits: int, n_layers: int, entanglement: str = 'linear') -> qml.QNode:
        """
        Create randomized quantum circuit.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers
            entanglement: Entanglement pattern
        
        Returns:
            qml.QNode: PennyLane quantum node
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev)
        def randomized_circuit(params):
            # Random initialization
            for i in range(n_qubits):
                qml.RX(np.random.uniform(0, 2*np.pi), wires=i)
                qml.RY(np.random.uniform(0, 2*np.pi), wires=i)
            
            # Randomized layers
            param_idx = 0
            for layer in range(n_layers):
                # Random single qubit rotations
                for i in range(n_qubits):
                    qml.RX(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RY(params[param_idx], wires=i)
                    param_idx += 1
                    qml.RZ(params[param_idx], wires=i)
                    param_idx += 1
                
                # Random entangling gates
                if entanglement == 'linear':
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                        qml.CRZ(params[param_idx], wires=[i, i+1])
                        param_idx += 1
                
                elif entanglement == 'circular':
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i+1) % n_qubits])
                        qml.CRZ(params[param_idx], wires=[i, (i+1) % n_qubits])
                        param_idx += 1
                
                elif entanglement == 'all_to_all':
                    for i, j in itertools.combinations(range(n_qubits), 2):
                        qml.CNOT(wires=[i, j])
                        qml.CRZ(params[param_idx], wires=[i, j])
                        param_idx += 1
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        return randomized_circuit
    
    def get_circuit(self, circuit_type: str, n_qubits: int, complexity: str) -> Tuple[qml.QNode, int]:
        """
        Get a quantum circuit based on type, qubit count, and complexity.
        
        Args:
            circuit_type: Type of circuit ('VUCCA', 'UCCA', 'IQP', 'Randomized')
            n_qubits: Number of qubits
            complexity: Complexity level ('low', 'medium', 'high', 'extreme')
        
        Returns:
            Tuple[qml.QNode, int]: Circuit and number of parameters
        """
        config = self.complexity_levels[complexity]
        n_layers = config['layers']
        entanglement = config['entanglement']
        
        if circuit_type == 'VUCCA':
            circuit = self.create_vucca_circuit(n_qubits, n_layers, entanglement)
        elif circuit_type == 'UCCA':
            circuit = self.create_ucca_circuit(n_qubits, n_layers, entanglement)
        elif circuit_type == 'IQP':
            circuit = self.create_iqp_circuit(n_qubits, n_layers, entanglement)
        elif circuit_type == 'Randomized':
            circuit = self.create_randomized_circuit(n_qubits, n_layers, entanglement)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Calculate number of parameters
        n_params = self._calculate_parameters(circuit_type, n_qubits, n_layers, entanglement)
        
        return circuit, n_params
    
    def _calculate_parameters(self, circuit_type: str, n_qubits: int, n_layers: int, entanglement: str) -> int:
        """
        Calculate the number of parameters for a given circuit configuration.
        
        Args:
            circuit_type: Type of circuit
            n_qubits: Number of qubits
            n_layers: Number of layers
            entanglement: Entanglement pattern
        
        Returns:
            int: Number of parameters
        """
        # Single qubit parameters per layer
        if circuit_type in ['VUCCA', 'Randomized']:
            single_qubit_params = 3 * n_qubits  # RX, RY, RZ
        elif circuit_type == 'UCCA':
            single_qubit_params = 2 * n_qubits  # RX, RY
        elif circuit_type == 'IQP':
            single_qubit_params = n_qubits  # RZ only
        
        # Entangling parameters per layer
        if entanglement == 'linear':
            entangling_params = n_qubits - 1
        elif entanglement == 'circular':
            entangling_params = n_qubits
        elif entanglement == 'all_to_all':
            entangling_params = n_qubits * (n_qubits - 1) // 2
        
        # Adjust for circuit type
        if circuit_type in ['VUCCA', 'UCCA']:
            entangling_params *= 2  # CRZ + CRY for UCCA, CRZ for VUCCA
        elif circuit_type == 'IQP':
            entangling_params *= 1  # CRZ only
        
        total_params = n_layers * (single_qubit_params + entangling_params)
        
        # Add initialization parameters for randomized circuits
        if circuit_type == 'Randomized':
            total_params += 2 * n_qubits  # Initial RX, RY
        
        return total_params
    
    def get_all_circuits(self, qubit_range: range, complexity_levels: List[str] = None) -> Dict:
        """
        Get all circuits for the specified qubit range and complexity levels.
        
        Args:
            qubit_range: Range of qubit counts
            complexity_levels: List of complexity levels to include
        
        Returns:
            Dict: Dictionary of all circuits
        """
        if complexity_levels is None:
            complexity_levels = list(self.complexity_levels.keys())
        
        circuits = {}
        
        for n_qubits in qubit_range:
            for circuit_type in self.circuit_types:
                for complexity in complexity_levels:
                    key = f"{circuit_type}_{n_qubits}q_{complexity}"
                    circuit, n_params = self.get_circuit(circuit_type, n_qubits, complexity)
                    circuits[key] = {
                        'circuit': circuit,
                        'n_params': n_params,
                        'n_qubits': n_qubits,
                        'circuit_type': circuit_type,
                        'complexity': complexity
                    }
        
        return circuits

# Example usage
if __name__ == "__main__":
    cd = CircuitDictionary()
    
    # Get a specific circuit
    circuit, n_params = cd.get_circuit('VUCCA', 4, 'medium')
    print(f"VUCCA circuit with 4 qubits, medium complexity: {n_params} parameters")
    
    # Get all circuits for qubit range 2-6
    all_circuits = cd.get_all_circuits(range(2, 7), ['low', 'medium'])
    print(f"Total circuits generated: {len(all_circuits)}")
    
    for key, info in all_circuits.items():
        print(f"{key}: {info['n_params']} parameters") 