import pennylane as qml

def generator_circuit_fn(params, noise, n_qubits, n_layers):
    """
    Quantum circuit for the generator.
    This function defines the structure of the quantum circuit.
    
    Args:
        params (array): Circuit parameters (weights).
        noise (array): Classical noise input.
        n_qubits (int): Total number of qubits.
        n_layers (int): Number of layers in the circuit.
    """
    # Encode the noise input
    for i in range(n_qubits):
        qml.RY(noise[i], wires=i)
    
    # Apply parameterized quantum layers
    param_idx = 0
    for _ in range(n_layers):
        # Rotation layer
        for i in range(n_qubits):
            qml.RX(params[param_idx], wires=i)
            param_idx += 1
            qml.RY(params[param_idx], wires=i)
            param_idx += 1
            qml.RZ(params[param_idx], wires=i)
            param_idx += 1
        
        # Entangling layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)] 