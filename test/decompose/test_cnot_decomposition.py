import rqcopt_mpo.jax_config

import numpy as np
import jax.numpy as jnp
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from rqcopt_mpo.circuit.cnot_decompose.cnot_circuit_builder import (
    cnot_decompose_circuit,
)
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
    trotterized_heisenberg_circuit,
)


def _unitary(circ: Circuit) -> np.ndarray:
    """Return the dense matrix of a circuit as a NumPy array."""
    return np.asarray(circ.to_matrix())


def _assert_unitaries_match(u1: np.ndarray, u2: np.ndarray, atol: float = 1e-10):
    """Assert equality up to a global phase."""
    idx = np.argmax(np.abs(u2))
    if np.isclose(u2.flat[idx], 0.0):
        idx = np.argmax(np.abs(u1))
        if np.isclose(u1.flat[idx], 0.0):
            raise AssertionError("Both unitaries are close to zero everywhere.")
    phase = np.angle(u1.flat[idx] / u2.flat[idx])
    np.testing.assert_allclose(u1, np.exp(1j * phase) * u2, atol=atol)


def _build_reference_circuit() -> Circuit:
    return trotterized_heisenberg_circuit(
        n_sites=8,
        J=1.0,
        D=1.0,
        h=1.0,
        dt=0.1,
        reps=4,
        order=2,
        dtype=jnp.complex128,
    )


def test_cnot_decomposition_unitaries_match():
    original = _build_reference_circuit()
    decomposed = cnot_decompose_circuit(original)

    assert decomposed.num_layers == original.num_layers + 1
    assert decomposed.layers[0].layer_index == 0
    assert all(g.is_single_qubit() for g in decomposed.layers[0].gates)

    orig_u = _unitary(original)
    decomp_u = _unitary(decomposed)
    _assert_unitaries_match(orig_u, decomp_u)


def test_single_two_qubit_gate_is_preserved():
    matrix = np.asarray(Operator(CXGate()).data)
    gate = Gate(
        matrix=matrix,
        qubits=(0, 1),
        layer_index=0,
        name="CX",
    )
    layer = GateLayer(layer_index=0, is_odd=False, gates=[gate], n_sites=2)
    circuit = Circuit(n_sites=2, dtype=jnp.complex128, layers=[layer])

    decomposed = cnot_decompose_circuit(circuit)

    assert len(decomposed.layers) == 2
    first_layer, second_layer = decomposed.layers
    assert first_layer.layer_index == 0
    assert second_layer.layer_index == 1
    assert all(g.is_single_qubit() for g in first_layer.gates)
    assert all(g.is_two_qubit() for g in second_layer.gates)

    _assert_unitaries_match(_unitary(circuit), _unitary(decomposed))
