# test_hilbert_schmidt_mpo.py
import pytest
import numpy as np
import jax.numpy as jnp

# Ensure JAX is configured (dtype, etc.) the same way as your project
import rqcopt_mpo.jax_config  # noqa: F401

from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo


def _hs_inner_product_from_mpo(A_mpo, B_mpo):
    """
    Compute Tr(A^\dagger B) where A_mpo, B_mpo are MPOs with site tensors shaped (Dl, d_up, d_down, Dr).
    """
    # Conjugate transpose A (dagger) at the MPO level
    A_dag = A_mpo.dagger()

    # Left environment starts as (1,1)
    L_env = jnp.eye(1).reshape(1, 1)

    # Contract site by site.
    # Einsum indices follow the original:
    # L_env: (a,e), A[i]: (a,b,c,d), B[i]: (e,c,b,f) -> new env (d,f)
    for i in range(len(A_dag)):
        L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, A_dag[i], B_mpo[i])

    # scalar overlap
    return L_env[0, 0]


@pytest.mark.parametrize(
    "n_sites, seed_A, seed_B, n_layers, p_single, p_two",
    [
        (6, 42, 43, 3, 0.3, 0.3),
        # add more quick cases if you like:
        # (3, 1, 2, 2, 0.5, 0.5),
    ],
)
def test_hilbert_schmidt_product_mpo_matches_matrix(n_sites, seed_A, seed_B, n_layers, p_single, p_two):
    # ---- Build circuit A ----
    A_circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers,
        p_single=p_single,
        p_two=p_two,
        seed=seed_A,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    A_circuit.sort_layers()
    A_mpo = circuit_to_mpo(A_circuit)
    A_mpo.left_canonicalize()
    A_matrix = np.asarray(A_circuit.to_matrix())

    # ---- Build circuit B ----
    B_circuit = generate_random_circuit(
        n_sites=n_sites,
        n_layers=n_layers,
        p_single=p_single,
        p_two=p_two,
        seed=seed_B,
        gate_name_single="U1",
        gate_name_two="U2",
        dtype=jnp.complex128,
    )
    B_circuit.sort_layers()
    B_mpo = circuit_to_mpo(B_circuit)
    B_mpo.left_canonicalize()
    B_matrix = np.asarray(B_circuit.to_matrix())

    # ---- Explicit Hilbert–Schmidt inner product: Tr(A^\dagger B) ----
    explicit = np.trace(np.conjugate(A_matrix).T @ B_matrix)

    # ---- MPO contraction version ----
    mpo_val = _hs_inner_product_from_mpo(A_mpo, B_mpo)
    mpo_val = np.asarray(mpo_val)  # convert JAX array to NumPy scalar

    # ---- Assertions ----
    # 1) Numerical equality
    np.testing.assert_allclose(mpo_val, explicit, rtol=1e-9, atol=1e-9)

    # 2) Sanity checks: scalar-like and finite
    assert np.ndim(mpo_val) == 0
    assert np.isfinite(mpo_val)

