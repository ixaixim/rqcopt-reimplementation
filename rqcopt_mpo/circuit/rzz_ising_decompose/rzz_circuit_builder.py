import rqcopt_mpo.jax_config

import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from qiskit.synthesis import TwoQubitWeylDecomposition
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

    for idx, layer in enumerate(sorted_layers):
        if not layer.is_odd: 
            for gate in layer.iterate_gates():
                weyl_decomp = TwoQubitWeylDecomposition(gate.matrix)
                a, b, c = weyl_decomp.a, weyl_decomp.b, weyl_decomp.c
                qc = QuantumCircuit(2)
                qc.append(UnitaryGate(weyl_decomp.K2l), [1]) 
                qc.append(UnitaryGate(weyl_decomp.K2r), [0]) 
                if abs(a) > 1e-10:
                    qc.h([0, 1])
                    qc.rzz(-2 * a, 0, 1) 
                    qc.h([0, 1])
                if abs(b) > 1e-10:
                    qc.rx(np.pi/2, [0, 1])
                    qc.rzz(-2 * b, 0, 1)
                    qc.rx(-np.pi/2, [0, 1]) 
                if abs(c) > 1e-10: # should be always ZERO
                    qc.rzz(-2 * c, 0, 1)
                qc.append(UnitaryGate(weyl_decomp.K1l), [1])
                qc.append(UnitaryGate(weyl_decomp.K1r), [0])
                # keep in mind Qiskit little endian convention: |q1q0> where the the most relevant bitstring is on the left.
                #  It will first iterate through q0 (for us is the less significant bit, what would be our q1 instead. And viceversa)
                # our notation is instead big endian: |q0q1>
                final_circ = transpile(qc, basis_gates=['rzz', 'u3'], optimization_level=3)
                
                params = []
                meta_list = []
                for instr in final_circ.data:
                    op = instr.operation
                    qargs = instr.qubits
                    # In Qiskit 1.0+, find_bit is the way. 
                    # But for simplicity, we can assume q.index if it's there
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
                
                # Update gate.matrix to exactly match the synthesized parameters
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)

        if layer.is_odd:
            # store the trotter angle 
            for gate in layer.iterate_gates():
                angle = rzz_angle(gate.matrix)
                gate.params = (np.array([angle]),)
                gate.name = "Ising_no_field"
                from rqcopt_mpo.circuit.noise import update_gate_matrix_from_params
                update_gate_matrix_from_params(gate, dtype=new_circ.dtype)
    return new_circ



    # if layer even:
    # iterate through gate: 
    # KAK decomposition. every gate is parametrizable 
    # odd layer: 
    # iterate through gate: extract angle to get the term: 2 * J * t_trotter (use function)
