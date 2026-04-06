import numpy as np
from qiskit import QuantumCircuit
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


def _append_zyz(qc: QuantumCircuit, phi: float, theta: float, lam: float, qubit: int):
    """
    Append Rz(phi) · Ry(theta) · Rz(lam) to *qc* using native Qiskit gates.

    We avoid ``qc.u(theta, phi, lam)`` because Qiskit's U gate carries an
    extra global phase  e^{i(phi+lam)/2}  relative to the standard ZYZ
    product.  That global phase becomes a *physically meaningful relative
    phase* when several single-qubit gates act on different qubits inside
    the same circuit.

    Using explicit ``qc.rz`` / ``qc.ry`` calls produces the exact matrix
    Rz(phi) @ Ry(theta) @ Rz(lam)  with the standard half-angle convention,
    matching our internal  ``_compose_k_from_zyz``.

    NOTE: The first matrix in the product (Rz(lam)) must be applied FIRST
    to the state, so it is appended first to the QuantumCircuit.
    """
    qc.rz(lam,   qubit)
    qc.ry(theta, qubit)
    qc.rz(phi,   qubit)


def circuit_to_qiskit(circuit: Circuit) -> QuantumCircuit:
    """
    Converts our custom Circuit dataclass to a qiskit.QuantumCircuit.
    Supports specific parametrized gate types:
    - Ising_hw_1q
    - Ising_hw_1rzz
    - Ising_no_field
    - Ising_synthesized

    NOTE: We map our site index `i` to Qiskit qubit index `n_sites - 1 - i`
    to match our Big-Endian (tensor-product order) convention.
    """
    n = circuit.n_sites
    qc = QuantumCircuit(n)

    def _q(i): return n - 1 - i

    # Ensure layers are sorted
    sorted_layers = sorted(circuit.layers, key=lambda layer: layer.layer_index)

    for layer in sorted_layers:
        # iterate_gates yields gates sorted by first qubit index
        for gate in layer.iterate_gates():
            if gate.name == "Ising_hw_1q":
                # params = ([phi, theta, lam],)
                # Our _get_euler_angles returns [phi, theta, lam]
                # Internal convention: K = Rz(phi) @ Ry(theta) @ Rz(lam)
                p = np.array(gate.params[0]).flatten()
                phi, theta, lam = float(p[0]), float(p[1]), float(p[2])
                _append_zyz(qc, phi, theta, lam, _q(gate.qubits[0]))

            elif gate.name == "Ising_hw_1rzz" or gate.name == "Ising_no_field":
                # params = ([angle],)
                theta = gate.params[0]
                if isinstance(theta, np.ndarray) or hasattr(theta, "item"):
                    theta = theta.item()
                qc.rzz(float(theta), _q(gate.qubits[0]), _q(gate.qubits[1]))

            elif gate.name == "Ising_synthesized":
                meta_list = gate.params_dict["metadata"]
                for i, meta in enumerate(meta_list):
                    op_name = meta["name"]
                    q_indices = meta["qubits"]
                    # Map relative qubit indices back to absolute circuit qubits.
                    # In rzz_decompose_ising_circuit, qc = QuantumCircuit(2) was used
                    # with qubit 1 as Left (site i) and qubit 0 as Right (site i+1).
                    # gate.qubits is (i, i+1).
                    # So qc qubit 1 -> gate.qubits[0]
                    #    qc qubit 0 -> gate.qubits[1]
                    # This is q -> 1 - q.
                    abs_qubits = [_q(gate.qubits[1 - q]) for q in q_indices]

                    p = gate.params[i]
                    if op_name == "u3":
                        # Qiskit transpile stores u3 params as (theta, phi, lam)
                        # matching Qiskit U-gate ordering.  The *matrix* stored
                        # during synthesis used  _compose_k_from_zyz(phi, theta, lam)
                        # so we call:  Rz(phi) Ry(theta) Rz(lam)
                        theta_val, phi_val, lam_val = float(p[0]), float(p[1]), float(p[2])
                        _append_zyz(qc, phi_val, theta_val, lam_val, abs_qubits[0])
                    elif op_name == "rzz":
                        qc.rzz(float(p[0]), abs_qubits[0], abs_qubits[1])
                    else:
                        raise ValueError(
                            f"Unsupported sub-gate in Ising_synthesized: {op_name}"
                        )
            else:
                raise ValueError(
                    f"Unsupported gate name for Qiskit export: {gate.name}"
                )

    return qc
