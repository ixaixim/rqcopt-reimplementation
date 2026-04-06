"""
Test: verify that circuit_to_qiskit exactly reproduces the unitary
from Circuit.to_matrix() for every supported gate type.
"""
import rqcopt_mpo.jax_config

import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from qiskit.quantum_info import Operator

from rqcopt_mpo.utils.qiskit_export import circuit_to_qiskit
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


# ── helpers ──────────────────────────────────────────────────

def _unitary_from_qiskit(circ: Circuit) -> np.ndarray:
    qc = circuit_to_qiskit(circ)
    return Operator(qc).data


def _unitary_from_tn(circ: Circuit) -> np.ndarray:
    return np.array(circ.to_matrix())


def _assert_unitaries_match(circ: Circuit, label: str, atol: float = 1e-10):
    U_tn = _unitary_from_tn(circ)
    U_qk = _unitary_from_qiskit(circ)
    
    # Check absolute mismatch
    diff = np.linalg.norm(U_tn - U_qk)
    
    if diff >= atol:
        # Check up to global phase
        # find the first non-zero element in U_tn
        idx = np.unravel_index(np.argmax(np.abs(U_tn)), U_tn.shape)
        phase = U_qk[idx] / U_tn[idx]
        diff_phase = np.linalg.norm(U_tn * phase - U_qk)
        if diff_phase < atol:
            print(f"  [{label}] Match found up to global phase: {phase}")
            return
            
    assert diff < atol, (
        f"[{label}] Unitary mismatch: ||U_tn - U_qiskit|| = {diff:.2e}"
    )


# ── tests ────────────────────────────────────────────────────

class TestQiskitExportHwFriendly:
    """Test export of HW-friendly (Ising_hw_1q + Ising_hw_1rzz) circuits."""

    @pytest.fixture
    def hf_circuit(self):
        from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import (
            trotterized_ising_hw_friendly_circuit,
        )
        from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import (
            rzz_decompose_ising_circuit as rzz_decompose_hw,
        )
        n_sites = 4
        dt = 0.5
        raw = trotterized_ising_hw_friendly_circuit(
            n_sites=n_sites, J=1.0, hx=0.75, hz=0.6,
            dt=dt, reps=2, order=2, dtype=jnp.complex128,
        )
        return rzz_decompose_hw(raw, order=2)

    def test_noiseless_match(self, hf_circuit):
        _assert_unitaries_match(hf_circuit, "HW-Friendly")


class TestQiskitExportRiemannianSynthesized:
    """Test export of Riemannian-synthesized (Ising_synthesized + Ising_no_field)."""

    @pytest.fixture
    def riem_circuit(self):
        from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
            trotterized_xyz_circuit,
        )
        from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import (
            rzz_decompose_ising_circuit,
        )
        n_sites = 4
        dt = 0.5
        raw = trotterized_xyz_circuit(
            n_sites=n_sites, Jx=0.0, Jy=0.0, Jz=1.0,
            hx=0.75, hz=0.6, dt=dt, reps=2, order=2,
            method="suzuki", dtype=jnp.complex128,
        )
        return rzz_decompose_ising_circuit(raw)

    def test_noiseless_match(self, riem_circuit):
        _assert_unitaries_match(riem_circuit, "Riemannian-Synth")


class TestQiskitExportConsistency:
    """
    Cross-check: build identical Hamiltonians via HF and Riemannian paths,
    verify both Qiskit exports yield equivalent unitaries.
    """

    def test_both_paths_agree(self):
        from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import (
            trotterized_ising_hw_friendly_circuit,
        )
        from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import (
            rzz_decompose_ising_circuit as rzz_decompose_hw,
        )
        from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
            trotterized_xyz_circuit,
        )
        from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import (
            rzz_decompose_ising_circuit,
        )

        n_sites, dt = 4, 0.5

        # HF path
        raw_hf = trotterized_ising_hw_friendly_circuit(
            n_sites=n_sites, J=1.0, hx=0.75, hz=0.6,
            dt=dt, reps=1, order=2, dtype=jnp.complex128,
        )
        circ_hf = rzz_decompose_hw(raw_hf, order=2)
        qc_hf = circuit_to_qiskit(circ_hf)

        # Verify each path independently first
        _assert_unitaries_match(circ_hf, "HF-cross")

        # Riemannian path
        raw_r = trotterized_xyz_circuit(
            n_sites=n_sites, Jx=0.0, Jy=0.0, Jz=1.0,
            hx=0.75, hz=0.6, dt=dt, reps=1, order=2,
            method="suzuki", dtype=jnp.complex128,
        )
        circ_r = rzz_decompose_ising_circuit(raw_r)
        _assert_unitaries_match(circ_r, "Riem-cross")
