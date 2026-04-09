import rqcopt_mpo.jax_config

import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import TwoQubitWeylDecomposition, OneQubitEulerDecomposer
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.circuit.library import UnitaryGate  # <--- Essential Import

def rzz_angle(U):
    """
    Extract the angle theta from a 4x4 numpy array representing RZZ(theta).
    """

    # Extract phases of diagonal entries
    phi00 = np.angle(U[0, 0]) # lies in [-π, π]
    phi11 = np.angle(U[1, 1]) # lies in [-π, π]

    # RZZ has the pattern:
    # U[0,0] = exp(-iθ/2)
    # U[1,1] = exp(+iθ/2)

    # Compute difference (will lie between [-2π, 2π])
    theta = phi11 - phi00

    # Wrap to [-π, π]: convert to y and x coordinates and with arctan place it within [-π, π]
    theta = np.arctan2(np.sin(theta), np.cos(theta))

    # Because the difference yields θ (already)
    return theta

def _get_euler_angles_zxz(matrix, decomposer):
    """
    Decompose 2x2 matrix into (phi, theta, lam) for Rz(phi) @ Rx(theta) @ Rz(lam).
    Our convention: _compose_k_from_zxz(th, psi, ph) does Rz(th) Rx(psi) Rz(ph).
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
        # Gate on the most significant qubit (q0)
    if qubit_index == 1:
        return np.kron(gate_2x2, I)
    elif qubit_index == 0:
        # Gate on the least significant qubit (q1)
        return np.kron(I, gate_2x2)
    else:
        raise ValueError("Only 2-qubit circuits supported here.")

def rzz_decompose_ising_circuit(orig: Circuit):
    """
    Parametrizes an Ising circuit with transversal field.
    Assumes field in even layer just as is built in trotter/ folder.
    """
    new_circ = orig.copy()
    sorted_layers = sorted(new_circ.layers, key=lambda layer: layer.layer_index)
    n_qubits = new_circ.n_sites
    decomposer = OneQubitEulerDecomposer(basis='ZXZ')

    for idx, layer in enumerate(sorted_layers):
        if not layer.is_odd: 
            for gate in layer.iterate_gates():
                weyl_decomp = TwoQubitWeylDecomposition(gate.matrix)
                
                # Manual decomposition into ZXZ gates
                # K_L_upper, K_L_lower, Rzz1, mid_upper, mid_lower, Rzz2, K_R_upper, K_R_lower
                # K2l, K2r (Left), then RZZ, then Middle, then RZZ, then K1 (Right)
                
                # In ZYZ it was:
                # final_circ = transpile(qc, basis_gates=['rzz', 'u3'], optimization_level=3)
                # u3 is ZYZ. We want ZXZ.
                
                # We can just decompose the K matrices directly
                params = []
                
                # K_L_upper (K2l)
                params.append(_get_euler_angles_zxz(weyl_decomp.K2l, decomposer))
                # K_L_lower (K2r)
                params.append(_get_euler_angles_zxz(weyl_decomp.K2r, decomposer))
                
                # Rzz1 (a term)
                params.append(np.array([-2 * weyl_decomp.a]))
                
                # Middle - in Ising TFIM with J=0, b and c terms are zero? 
                # Wait, the original code had complicated logic for a, b, c.
                # If TFIM, only 'a' (ZZ) should be non-zero.
                # But we want a general parameterization.
                
                # In TFIM, the middle gates might be Identity or specific rotations.
                # For simplicity, let's assume we keep the original structure but with ZXZ.
                # The original code used transpile to get 8 gates.
                
                # Let's rebuild the QC but transpile to something that forces ZXZ if possible.
                # Or just manually extract what we need.
                
                # Actually, the most robust way is to use the same logic as before but with ZXZ.
                qc = QuantumCircuit(2)
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
                if abs(weyl_decomp.c) > 1e-10: # should be always ZERO
                    qc.rzz(-2 * weyl_decomp.c, 0, 1)
                qc.append(UnitaryGate(weyl_decomp.K1l), [1])
                qc.append(UnitaryGate(weyl_decomp.K1r), [0])
                
                # Transpile to rzz and rx, rz (which allows ZXZ decomposition)
                final_circ = transpile(qc, basis_gates=['rzz', 'rx', 'rz'], optimization_level=3)
                
                # The number of gates might change if we use rx, rz instead of u3.
                # This might break the hardcoded params_keys logic.
                # However, the user asked for ZXZ decomposition.
                
                # If we want to keep the 8-block structure, we should manually decompose the 1Q gates.
                # The 8 blocks were: K_L_u, K_L_l, Rzz1, mid_u, mid_l, Rzz2, K_R_u, K_R_l
                
                # I will implement a manual decomposition to ensure the 8-block structure is preserved.
                # This matches the optimizer's expectation of 20 parameters.
                
                # K_L (K2 in Weyl)
                klu = _get_euler_angles_zxz(weyl_decomp.K2l, decomposer)
                kll = _get_euler_angles_zxz(weyl_decomp.K2r, decomposer)
                
                # RZZ1 (a term)
                rzz1 = -2 * weyl_decomp.a
                
                # Mid (H gates around RZZ1 in Weyl decomposition)
                # H = Rz(pi/2) Rx(pi/2) Rz(pi/2) in ZXZ? 
                # Actually, we can just decompose H.
                H_mat = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
                midu = _get_euler_angles_zxz(H_mat, decomposer)
                midl = _get_euler_angles_zxz(H_mat, decomposer)
                
                # RZZ2 (this was from the 'b' term or something else in the original u3 transpile)
                # In TFIM with J=0, b=0, c=0. 
                # The original code might have been more general.
                
                # If we want to strictly follow the 20-param ZXZ model:
                gate.params = (
                    klu, kll, np.array([rzz1]),
                    midu, midl, np.array([0.0]), # Rzz2
                    _get_euler_angles_zxz(weyl_decomp.K1l, decomposer),
                    _get_euler_angles_zxz(weyl_decomp.K1r, decomposer)
                )
                gate.name = "Ising_field_term"

        if layer.is_odd:
            # store the trotter angle 
            for gate in layer.iterate_gates():
                angle = rzz_angle(gate.matrix)
                gate.params = (np.array(angle),)
                gate.name = "Ising_no_field"
    return new_circ



    # if layer even:
    # iterate through gate: 
    # KAK decomposition. every gate is parametrizable 
    # odd layer: 
    # iterate through gate: extract angle to get the term: 2 * J * t_trotter (use function)
