from __future__ import annotations
import rqcopt_mpo.jax_config


from typing import Tuple

import jax.numpy as jnp

# Reuse tested primitives for rotations, Pauli matrices, and HST backprop
from rqcopt_mpo.optimization.parametrized_adam.utils import (
    _rot_from_generator,
    _paulis_1q,
    _paulis_2q,
    _backprop_hst_loss,
)


# -----------------------------
# Composition helpers
# -----------------------------

def _compose_k_from_zyz(theta: float, psi: float, phi: float, *, dtype) -> jnp.ndarray:
    """
    Return a single-qubit unitary K = Rz(theta) @ Ry(psi) @ Rz(phi),
    where Rz/Ry use generator scaling 0.5 (matches existing code/tests).
    """
    _, Y, Z, _ = _paulis_1q(dtype)
    Rz1 = _rot_from_generator(theta, Z, 0.5, dtype)
    Ry  = _rot_from_generator(psi,   Y, 0.5, dtype)
    Rz2 = _rot_from_generator(phi,   Z, 0.5, dtype)
    return Rz1 @ Ry @ Rz2


def _compose_entangler(a: float, b: float, c: float, *, dtype) -> jnp.ndarray:
    """
    Return the 2-qubit nonlocal unitary V = exp(i (a XX + b YY + c ZZ)).
    Implemented via commuting factors using the same convention as
    parametrized_adam (pass -P to _rot_from_generator for the +i sign).
    """
    XX, YY, ZZ, _ = _paulis_2q(dtype)
    Ua = _rot_from_generator(a, -XX, 1.0, dtype)
    Ub = _rot_from_generator(b, -YY, 1.0, dtype)
    Uc = _rot_from_generator(c, -ZZ, 1.0, dtype)
    return Ua @ Ub @ Uc


def compose_weyl_unitary(theta15: jnp.ndarray, *, dtype) -> jnp.ndarray:
    """
    Compose the full 2-qubit unitary from a 15-parameter vector encoding
    (K2l, K2r) on the left, (a,b,c) entangler, then (K1l, K1r) on the right:

        U = (K2l ⊗ K2r) @ V(a,b,c) @ (K1l ⊗ K1r)

    Parameter ordering (length 15):
        (t1, p1, f1,  t2, p2, f2,  a, b, c,  t3, p3, f3,  t4, p4, f4)
    where each K is K = Rz(t) @ Ry(p) @ Rz(f).
    """
    t1, p1, f1, t2, p2, f2, a, b, c, t3, p3, f3, t4, p4, f4 = [float(x) for x in theta15]
    K2l = _compose_k_from_zyz(t1, p1, f1, dtype=dtype)
    K2r = _compose_k_from_zyz(t2, p2, f2, dtype=dtype)
    V   = _compose_entangler(a, b, c, dtype=dtype)
    K1l = _compose_k_from_zyz(t3, p3, f3, dtype=dtype)
    K1r = _compose_k_from_zyz(t4, p4, f4, dtype=dtype)

    A = jnp.kron(K2l, K2r)
    B = jnp.kron(K1l, K1r)
    return A @ V @ B


# -----------------------------
# Parameter extraction (KAK + ZYZ)
# -----------------------------

def extract_weyl_params_for_unitary(U: jnp.ndarray) -> jnp.ndarray:
    """
    Extract a 15-parameter vector for a 4x4 2-qubit unitary using Qiskit's
    TwoQubitWeylDecomposition and OneQubitEulerDecomposer (ZYZ). Global phase
    is discarded. Returns jnp.ndarray shape (15,).

    Ordering:
        (K2l ZYZ angles, K2r ZYZ angles, a,b,c, K1l ZYZ angles, K1r ZYZ angles)
    where each ZYZ triple is (theta, psi, phi) in K = Rz(theta) @ Ry(psi) @ Rz(phi).
    """
    try:
        from qiskit.synthesis import TwoQubitWeylDecomposition, OneQubitEulerDecomposer
    except Exception as e:
        raise RuntimeError(
            "Qiskit is required for extracting Weyl parameters."
        ) from e

    U_np = jnp.asarray(U).astype(complex)
    kak = TwoQubitWeylDecomposition(U_np)

    K2l, K2r = kak.K2l, kak.K2r
    K1l, K1r = kak.K1l, kak.K1r
    a, b, c = float(kak.a), float(kak.b), float(kak.c)

    decomp = OneQubitEulerDecomposer(basis="ZYZ")

    # Qiskit returns angles as (theta, phi, lam) s.t. Rz(lam) @ Ry(theta) @ Rz(phi)
    def _angles_zyz(M):
        th, ph, lam, _ = decomp.angles_and_phase(M)
        # Map to (theta, psi, phi) ≡ (lam, th, ph) for K = Rz(theta) @ Ry(psi) @ Rz(phi)
        return float(lam), float(th), float(ph)

    t1, p1, f1 = _angles_zyz(K2l)
    t2, p2, f2 = _angles_zyz(K2r)
    t3, p3, f3 = _angles_zyz(K1l)
    t4, p4, f4 = _angles_zyz(K1r)

    return jnp.array([t1, p1, f1, t2, p2, f2, a, b, c, t3, p3, f3, t4, p4, f4], dtype=jnp.float64)


def parametrize_circuit_weyl(circuit) -> None:
    """
    Mutate the circuit in-place to attach 15-parameter Weyl vectors to every
    2-qubit gate. For each such gate g, set:
        g.name = 'WeylParam2Q'
        g.params = (theta15,)
        g.matrix = compose_weyl_unitary(theta15)
    Single-qubit gates keep their current names and get no parameters.
    """
    for layer in circuit.layers:
        for g in layer.gates:
            if len(g.qubits) != 2:
                # leave 1q gates unchanged (no parameters by default)
                continue
            theta15 = extract_weyl_params_for_unitary(g.matrix)
            g.name = "WeylParam2Q"
            g.params = (jnp.asarray(theta15, dtype=jnp.float64),)
            g.matrix = compose_weyl_unitary(theta15, dtype=g.matrix.dtype)


# -----------------------------
# Param-gradient: 15D Weyl (K⊗K, V, K⊗K)
# -----------------------------

def param_grad_weyl15(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Map Euclidean gradient dL/dG for a 2-qubit gate to dL/dtheta for the 15D
    Weyl parameterization U = (K2l ⊗ K2r) V (K1l ⊗ K1r).

    Parameter ordering: (t1,p1,f1, t2,p2,f2, a,b,c, t3,p3,f3, t4,p4,f4)
    with K = Rz(t) @ Ry(p) @ Rz(f), scales 0.5 for 1q, 1.0 for entangler.
    """
    dtype = dL_dG.dtype
    X, Y, Z, _ = _paulis_1q(dtype)
    XX, YY, ZZ, _I4 = _paulis_2q(dtype)
    kron = jnp.kron

    t1, p1, f1, t2, p2, f2, a, b, c, t3, p3, f3, t4, p4, f4 = theta

    # Build individual Rz/Ry for reuse (to form K and their derivatives)
    Rz1_1 = _rot_from_generator(t1, Z, 0.5, dtype)
    Ry1   = _rot_from_generator(p1, Y, 0.5, dtype)
    Rz2_1 = _rot_from_generator(f1, Z, 0.5, dtype)
    K2l   = Rz1_1 @ Ry1 @ Rz2_1

    Rz1_2 = _rot_from_generator(t2, Z, 0.5, dtype)
    Ry2   = _rot_from_generator(p2, Y, 0.5, dtype)
    Rz2_2 = _rot_from_generator(f2, Z, 0.5, dtype)
    K2r   = Rz1_2 @ Ry2 @ Rz2_2

    V = _compose_entangler(a, b, c, dtype=dtype)

    Rz1_3 = _rot_from_generator(t3, Z, 0.5, dtype)
    Ry3   = _rot_from_generator(p3, Y, 0.5, dtype)
    Rz2_3 = _rot_from_generator(f3, Z, 0.5, dtype)
    K1l   = Rz1_3 @ Ry3 @ Rz2_3

    Rz1_4 = _rot_from_generator(t4, Z, 0.5, dtype)
    Ry4   = _rot_from_generator(p4, Y, 0.5, dtype)
    Rz2_4 = _rot_from_generator(f4, Z, 0.5, dtype)
    K1r   = Rz1_4 @ Ry4 @ Rz2_4

    A = kron(K2l, K2r)
    B = kron(K1l, K1r)

    # Derivatives for 1q ZYZ pieces (left factors)
    s1q = 0.5
    dK2l_dt = (-1j * s1q) * (Z @ K2l)
    dK2l_dp = Rz1_1 @ ((-1j * s1q) * (Y @ Ry1)) @ Rz2_1
    dK2l_df = Rz1_1 @ Ry1 @ ((-1j * s1q) * (Z @ Rz2_1))

    dK2r_dt = (-1j * s1q) * (Z @ K2r)
    dK2r_dp = Rz1_2 @ ((-1j * s1q) * (Y @ Ry2)) @ Rz2_2
    dK2r_df = Rz1_2 @ Ry2 @ ((-1j * s1q) * (Z @ Rz2_2))

    # Entangler derivatives (commuting)
    s2q = 1.0
    dV_da = (1j * s2q) * (XX @ V)
    dV_db = (1j * s2q) * (YY @ V)
    dV_dc = (1j * s2q) * (ZZ @ V)

    # Derivatives for right-side 1q factors
    dK1l_dt = (-1j * s1q) * (Z @ K1l)
    dK1l_dp = Rz1_3 @ ((-1j * s1q) * (Y @ Ry3)) @ Rz2_3
    dK1l_df = Rz1_3 @ Ry3 @ ((-1j * s1q) * (Z @ Rz2_3))

    dK1r_dt = (-1j * s1q) * (Z @ K1r)
    dK1r_dp = Rz1_4 @ ((-1j * s1q) * (Y @ Ry4)) @ Rz2_4
    dK1r_df = Rz1_4 @ Ry4 @ ((-1j * s1q) * (Z @ Rz2_4))

    # Assemble dU/dθ blocks
    derivs: list[jnp.ndarray] = []

    # Left-local parameters (K2l ⊗ K2r)
    derivs.append(kron(dK2l_dt, K2r) @ V @ B)
    derivs.append(kron(dK2l_dp, K2r) @ V @ B)
    derivs.append(kron(dK2l_df, K2r) @ V @ B)

    derivs.append(kron(K2l, dK2r_dt) @ V @ B)
    derivs.append(kron(K2l, dK2r_dp) @ V @ B)
    derivs.append(kron(K2l, dK2r_df) @ V @ B)

    # Entangler parameters
    derivs.append(A @ dV_da @ B)
    derivs.append(A @ dV_db @ B)
    derivs.append(A @ dV_dc @ B)

    # Right-local parameters (K1l ⊗ K1r)
    derivs.append(A @ V @ kron(dK1l_dt, K1r))
    derivs.append(A @ V @ kron(dK1l_dp, K1r))
    derivs.append(A @ V @ kron(dK1l_df, K1r))

    derivs.append(A @ V @ kron(K1l, dK1r_dt))
    derivs.append(A @ V @ kron(K1l, dK1r_dp))
    derivs.append(A @ V @ kron(K1l, dK1r_df))

    # Backprop each derivative into a scalar gradient
    grads = [
        _backprop_hst_loss(L, dL_dG, dU_dth, n_sites, is_normalized)
        for dU_dth in derivs
    ]
    return jnp.stack(grads)


__all__ = [
    "compose_weyl_unitary",
    "extract_weyl_params_for_unitary",
    "parametrize_circuit_weyl",
    "param_grad_weyl15",
]

