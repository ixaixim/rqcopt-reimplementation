import rqcopt_mpo.jax_config

import numpy as np
import jax.numpy as jnp
import pytest
from qiskit.circuit.library import CXGate, CZGate, RZZGate
from qiskit.quantum_info import Operator, random_unitary

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer
from rqcopt_mpo.circuit.cnot_decompose.cnot_circuit_builder import (
    cnot_absorb_1q_gates, _decompose_gate_into_cnot_blocks
)
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
    trotterized_heisenberg_circuit,
)


def _unitary(circ: Circuit) -> np.ndarray:
    return np.asarray(circ.to_matrix())


def _assert_same_unitaries(u1: np.ndarray, u2: np.ndarray, atol: float = 1e-10):
    idx = np.argmax(np.abs(u2))
    if np.isclose(u2.flat[idx], 0.0):
        idx = np.argmax(np.abs(u1))
    phase = np.angle(u1.flat[idx] / u2.flat[idx])
    np.testing.assert_allclose(u1, np.exp(1j * phase) * u2, atol=atol)


def _build_reference_circuit() -> Circuit:
    return trotterized_heisenberg_circuit(
        n_sites=10,
        J=1.0,
        D=1.5,
        h=0.5,
        dt=0.3, # to do: try mulitples of pi/2,
        reps=1,
        order=2, # todo: try order 1 (i.e. ending in odd layer)
        dtype=jnp.complex128,
    )


@pytest.mark.parametrize(
    "unitary",
    [
        # CXGate().to_matrix(),
        # CZGate().to_matrix(),
        # RZZGate(0.3).to_matrix(),
        random_unitary(4, seed=1234).data,
        random_unitary(4, seed=5678).data,
    ],
)
def test_decompose_gate_into_cnot_blocks_reconstructs_unitary(unitary):
    gate = Gate(
        matrix=np.asarray(unitary, dtype=np.complex128),
        qubits=(0, 1),
        layer_index=0,
        name="test_gate",
    )

    pre, middle, post = _decompose_gate_into_cnot_blocks(gate, basis_fidelity=None)

    q0, q1 = gate.qubits
    pre_full = np.kron(pre[q0], pre[q1])
    post_full = np.kron(post[q0], post[q1])

    reconstructed = post_full @ middle @ pre_full
    _assert_same_unitaries(np.asarray(unitary, dtype=np.complex128), reconstructed)


# TODO: investigate why for specific time steps it does not work (multiple of pi/2 time steps) probably bc the gate gets decomposed into fewer cnots?
def test_cnot_absorb_1q_gates_preserves_unitary():
    """Check that absorbing 1q gates into 2q blocks leaves the circuit unitary unchanged."""
    orig = _build_reference_circuit()
    absorbed = cnot_absorb_1q_gates(orig, basis_fidelity=None)

    u_orig = _unitary(orig)
    u_absorbed = _unitary(absorbed)

    _assert_same_unitaries(u_orig, u_absorbed)
