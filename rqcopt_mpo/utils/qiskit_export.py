import numpy as np
from qiskit import QuantumCircuit
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit

def circuit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """
    Converts our custom Circuit dataclass to a qiskit.QuantumCircuit.
    Supports specific parametrized gate types:
    - Ising_hw_1q
    - Ising_hw_1rzz
    - Ising_no_field
    - Ising_synthesized
    """
    qc = QuantumCircuit(circuit.n_sites)
    
    # Ensure layers are sorted
    sorted_layers = sorted(circuit.layers, key=lambda layer: layer.layer_index)
    
    for layer in sorted_layers:
        # iterate_gates yields gates sorted by first qubit index
        for gate in layer.iterate_gates():
            if gate.name == "Ising_hw_1q":
                # params = ([phi, theta, lam],)
                # Note: Qiskit U gate expects (theta, phi, lam)
                # Our _get_euler_angles returns [phi, theta, lam]
                p = np.array(gate.params[0]).flatten()
                phi, theta, lam = p[0], p[1], p[2]
                qc.u(theta, phi, lam, gate.qubits[0])
                
            elif gate.name == "Ising_hw_1rzz" or gate.name == "Ising_no_field":
                # params = ([angle],)
                theta = gate.params[0]
                if isinstance(theta, np.ndarray) or hasattr(theta, "item"):
                    theta = theta.item()
                qc.rzz(theta, gate.qubits[0], gate.qubits[1])
                
            elif gate.name == "Ising_synthesized":
                meta_list = gate.params_dict["metadata"]
                for i, meta in enumerate(meta_list):
                    op_name = meta["name"]
                    q_indices = meta["qubits"]
                    # In Qiskit, q_indices are relative to the original 2-qubit space
                    # However, transpile may have returned logical indices or relative ones depending on how it was called
                    # Looking at rzz_circuit_builder.py, `qc = QuantumCircuit(2)` was used.
                    # So q_indices are [0, 1] (or just [0] or [1]).
                    # We must map these relative indices back to the absolute qubits of the gate.
                    # gate.qubits is (q_left, q_right)
                    abs_qubits = [gate.qubits[q] for q in q_indices]
                    
                    p = gate.params[i]
                    if op_name == "u3":
                        # In Qiskit transpile, u3 params are (theta, phi, lam)
                        theta, phi, lam = p[0], p[1], p[2]
                        qc.u(theta, phi, lam, abs_qubits[0])
                    elif op_name == "rzz":
                        theta = p[0]
                        qc.rzz(theta, abs_qubits[0], abs_qubits[1])
                    else:
                        raise ValueError(f"Unsupported sub-gate in Ising_synthesized: {op_name}")
            else:
                raise ValueError(f"Unsupported gate name for Qiskit export: {gate.name}")
                
    return qc
