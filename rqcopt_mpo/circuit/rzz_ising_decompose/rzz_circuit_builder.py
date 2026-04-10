import rqcopt_mpo.jax_config

import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import TwoQubitWeylDecomposition, OneQubitEulerDecomposer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import UnitaryGate

def rzz_angle_and_phase(U):
    """
    Extract the angle theta and global phase from a 4x4 numpy array representing RZZ(theta).
    U = exp(i*phase) * exp(-i*theta/2 * ZZ)
    U[0,0] = exp(i*phase) * exp(-i*theta/2)
    U[1,1] = exp(i*phase) * exp(i*theta/2)
    """
    phi00 = np.angle(U[0, 0])
    phi11 = np.angle(U[1, 1])
    theta = phi11 - phi00
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    # phase - theta/2 = phi00  => phase = phi00 + theta/2
    phase = phi00 + 0.5 * theta
    return theta, phase

def _get_euler_angles_zxz(matrix, decomposer):
    """
    Decompose 2x2 matrix into (phi, theta, lam) for Rz(phi) @ Rx(theta) @ Rz(lam).
    """
    if matrix is None:
        return np.zeros(3)
    theta, phi, lam = decomposer.angles(matrix)
    # Qiskit ZXZ: Rz(phi) Rx(theta) Rz(lam)
    # Our _compose_k_from_zxz(th, psi, ph): Rz(th) Rx(psi) Rz(ph)
    # So: th=phi, psi=theta, ph=lam
    return np.array([phi, theta, lam])

def embed_single_qubit_gate(gate_2x2, qubit_index):
    """Return 4x4 matrix of a 1-qubit gate embedded in 2-qubit space."""
    I = np.eye(2, dtype=complex)
    if qubit_index == 1:
        return np.kron(gate_2x2, I)
    elif qubit_index == 0:
        return np.kron(I, gate_2x2)
    else:
        raise ValueError("Only 2-qubit circuits supported here.")

def rzz_decompose_ising_circuit(orig: Circuit):
    """
    Synthesizes each 2-qubit gate into RZZ + RX + RZ gates (ZXZ basis).
    This version is flexible and used for robustness analysis.
    Uses 'Ising_synthesized' gate name.
    """
    new_circ = orig.copy()
    sorted_layers = sorted(new_circ.layers, key=lambda layer: layer.layer_index)
    
    for idx, layer in enumerate(sorted_layers):
        for gate in layer.iterate_gates():
            if gate.is_two_qubit():
                weyl_decomp = TwoQubitWeylDecomposition(gate.matrix)
                
                qc = QuantumCircuit(2)
                # Apply K2 gates first (closer to the input)
                qc.append(UnitaryGate(weyl_decomp.K2l), [1]) 
                qc.append(UnitaryGate(weyl_decomp.K2r), [0]) 
                if abs(weyl_decomp.a) > 1e-10:
                    qc.h([0, 1])
                    qc.rzz(-2 * weyl_decomp.a, 0, 1) 
                    qc.h([0, 1])
                if abs(weyl_decomp.b) > 1e-10:
                    qc.rx(np.pi/2, [0, 1])
                    qc.rzz(-2 * weyl_decomp.b, 0, 1)
                    qc.rx(-np.pi/2, [0, 1]) 
                if abs(weyl_decomp.c) > 1e-10:
                    qc.rzz(-2 * weyl_decomp.c, 0, 1)
                # Apply K1 gates last (closer to the output)
                qc.append(UnitaryGate(weyl_decomp.K1l), [1])
                qc.append(UnitaryGate(weyl_decomp.K1r), [0])
                
                # Force ZXZ by using rx, rz in basis_gates
                final_circ = transpile(qc, basis_gates=['rzz', 'rx', 'rz'], optimization_level=3)
                
                params = []
                meta_list = []
                # Add global phase to metadata
                params.append(np.array([final_circ.global_phase]))
                meta_list.append({"name": "global_phase", "qubits": []})

                for instr in final_circ.data:
                    op = instr.operation
                    qargs = instr.qubits
                    try:
                        q_indices = [q.index for q in qargs]
                    except AttributeError:
                        q_indices = [final_circ.find_bit(q).index for q in qargs]
                    
                    params.append(np.array(op.params))
                    meta_list.append({
                        "name": op.name,
                        "qubits": q_indices
                    })

                gate.params = tuple(params)
                gate.params_dict = {"metadata": meta_list}
                gate.name = "Ising_synthesized"
                
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)
            else:
                # For single-qubit gates, we can decompose into ZXZ directly
                decomposer = OneQubitEulerDecomposer(basis='ZXZ')
                euler_angles = _get_euler_angles_zxz(gate.matrix, decomposer)
                gate.params = (euler_angles,)
                gate.name = "Ising_hw_1q"
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)
                
    return new_circ

def rzz_decompose_ising_circuit_fixed_params(orig: Circuit):
    """
    Parametrizes each 2-qubit gate into a fixed 8-block structure using ZXZ rotations.
    This version is intended for the QNG/Adam optimizer.
    Uses 'Ising_field_term' gate name.
    """
    new_circ = orig.copy()
    sorted_layers = sorted(new_circ.layers, key=lambda layer: layer.layer_index)
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')

    for idx, layer in enumerate(sorted_layers):
        for gate in layer.iterate_gates():
            if gate.is_two_qubit():
                weyl_decomp = TwoQubitWeylDecomposition(gate.matrix)
                
                klu = _get_euler_angles_zxz(weyl_decomp.K2l, decomposer)
                kll = _get_euler_angles_zxz(weyl_decomp.K2r, decomposer)
                rzz1 = -2 * weyl_decomp.a
                
                # Mid (H gates around RZZ1)
                H_mat = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
                midu = _get_euler_angles_zxz(H_mat, decomposer)
                midl = _get_euler_angles_zxz(H_mat, decomposer)
                
                gate.params = (
                    klu, kll, np.array([rzz1]),
                    midu, midl, np.array([0.0]), # Rzz2
                    _get_euler_angles_zxz(weyl_decomp.K1l, decomposer),
                    _get_euler_angles_zxz(weyl_decomp.K1r, decomposer)
                )
                gate.name = "Ising_field_term"
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)
            else:
                # Handle single-qubit gates
                euler_angles = _get_euler_angles_zxz(gate.matrix, decomposer)
                gate.params = (euler_angles,)
                gate.name = "Ising_hw_1q"
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)
                
    return new_circ
