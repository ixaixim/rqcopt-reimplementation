import rqcopt_mpo.jax_config
import numpy as np
import jax.numpy as jnp
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import OneQubitEulerDecomposer

def rzz_angle(U):
    """
    Extract the angle theta from a 4x4 numpy array representing RZZ(theta).
    Assumes U is diagonal (pure ZZ interaction).
    """
    # Extract phases of diagonal entries
    phi00 = np.angle(U[0, 0]) # lies in [-π, π]
    phi11 = np.angle(U[1, 1]) # lies in [-π, π]

    # RZZ has the pattern: diag(exp(-iθ/2), exp(+iθ/2), exp(+iθ/2), exp(-iθ/2))
    # Difference phi11 - phi00 = (iθ/2) - (-iθ/2) = iθ (in phase) -> theta
    theta = phi11 - phi00

    # Wrap to [-π, π]
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    return theta

def _get_euler_angles(matrix, decomposer):
    """
    Decompose 2x2 matrix into (th, psi, ph) for Rz(th) @ Rx(psi) @ Rz(ph).
    Qiskit 'ZXZ' returns (theta, phi, lam) for Rz(phi) Rx(theta) Rz(lam).
    """
    if matrix is None:
        return np.zeros(3)
    
    # Qiskit returns (theta, phi, lam) for Rz(phi) Rx(theta) Rz(lam)
    theta, phi, lam = decomposer.angles(matrix)
    
    # Map to our convention: Rz(phi) Rx(theta) Rz(lam)
    # Our function _compose_k_from_zxz(th, psi, ph) does Rz(th) Rx(psi) Rz(ph)
    # So: th=phi, psi=theta, ph=lam
    return np.array([phi, theta, lam])

def _create_parametrized_gate(matrix, qubits, layer_idx, rzz_matrix, m1, m2, decomposer):
    """Creates a Gate with 'Ising_hw_1rzz' parametrization."""
    theta = rzz_angle(rzz_matrix)
    
    
    # Param structure: L_up, L_low, theta, R_up, R_low
    # Cast to arrays to ensure they are mutable/compatible with JAX pytrees later
    params = (
        np.array([theta]),
    )
    
    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="Ising_hw_1rzz",
        params=params
    )

def _create_parametrized_1q_gate(matrix, qubits, layer_idx, decomposer):
    """Creates a Gate with 'Ising_hw_1q' parametrization."""
    # Decompose the single qubit gate
    params_euler = _get_euler_angles(matrix, decomposer)
    
    return Gate(
        matrix=matrix,
        qubits=qubits,
        layer_index=layer_idx,
        name="Ising_hw_1q",
        params=(params_euler,)
    )

def rzz_decompose_ising_circuit(orig: Circuit, order: int = 2) -> Circuit:
    """
    Decomposes the circuit into 'Ising_hw_1rzz' and 'Ising_hw_1q' gates.
    Does NOT merge layers.
    2Q Layers -> Ising_hw_1rzz (with Identity locals).
    1Q Layers -> Ising_hw_1q.
    """
    sorted_layers = sorted(orig.layers, key=lambda layer: layer.layer_index)
    new_layers = []
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')
    
    for i, current_layer in enumerate(sorted_layers):
        new_gates = []
        
        # Check if the layer contains 2-qubit gates or 1-qubit gates
        # We assume uniform layers (all gates are same type roughly)
        if not current_layer.gates:
            # Empty layer, just copy structure
            new_layer = GateLayer(
                layer_index=len(new_layers),
                is_odd=current_layer.is_odd,
                gates=[],
                n_sites=current_layer.n_sites
            )
            new_layers.append(new_layer)
            continue
            
        first_gate = current_layer.gates[0]
        
        if len(first_gate.qubits) == 2:
            # Interaction Layer (2Q)
            for gate in current_layer.iterate_gates():
                # For pure decomposition without merging, locals are Identity
                m1 = np.eye(2, dtype=gate.matrix.dtype)
                m2 = np.eye(2, dtype=gate.matrix.dtype)
                
                new_gate = _create_parametrized_gate(
                    gate.matrix, gate.qubits, len(new_layers),
                    gate.matrix, m1, m2, decomposer
                )
                new_gates.append(new_gate)
                
        elif len(first_gate.qubits) == 1:
            # Field Layer (1Q)
            for gate in current_layer.iterate_gates():
                new_gate = _create_parametrized_1q_gate(
                    gate.matrix, gate.qubits, len(new_layers), decomposer
                )
                new_gates.append(new_gate)
        
        else:
            raise ValueError(f"Unsupported gate size: {len(first_gate.qubits)} qubits")

        new_layer = GateLayer(
            layer_index=len(new_layers),
            is_odd=current_layer.is_odd,
            gates=new_gates,
            n_sites=current_layer.n_sites
        )
        new_layers.append(new_layer)

    return Circuit(n_sites=orig.n_sites, dtype=orig.dtype, layers=new_layers)
