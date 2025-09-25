import jax.numpy as jnp
import numpy as np
import pytest

import rqcopt_mpo.jax_config  # ensure JAX dtype/config is initialized

from rqcopt_mpo.optimization.parametrized_adam.utils import (
    param_grad_rx,
    param_grad_ry,
    param_grad_rz,
    param_grad_xxyyzz,
)


def _paulis_1q(dtype):
    X = jnp.array([[0, 1], [1, 0]], dtype=dtype)
    Y = jnp.array([[0, -1j], [1j, 0]], dtype=dtype)
    Z = jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I = jnp.eye(2, dtype=dtype)
    return X, Y, Z, I


def _paulis_2q(dtype):
    X, Y, Z, I2 = _paulis_1q(dtype)
    kron = jnp.kron
    XX = kron(X, X)
    YY = kron(Y, Y)
    ZZ = kron(Z, Z)
    I4 = jnp.eye(4, dtype=dtype)
    return XX, YY, ZZ, I4


def _rot_from_generator(theta: float, P: jnp.ndarray, scale: float, dtype) -> jnp.ndarray:
    # exp(-i * scale * theta * P) for involutory P (P^2 = I)
    c = jnp.cos(scale * theta)
    s = jnp.sin(scale * theta)
    I = jnp.eye(P.shape[0], dtype=dtype)
    return c * I - 1j * s * P


def _dU_dtheta(theta: float, P: jnp.ndarray, scale: float, dtype) -> jnp.ndarray:
    U = _rot_from_generator(theta, P, scale, dtype)
    return (-1j * scale) * (P @ U)


@pytest.mark.parametrize("axis,fn_name", [("x", "rx"), ("y", "ry"), ("z", "rz")])
@pytest.mark.parametrize("is_normalized", [False, True])
def test_param_grad_1q_matches_analytic(axis, fn_name, is_normalized):
    # Setup: 1-qubit reference and parameterized single-qubit rotation
    dtype = jnp.complex128
    X, Y, Z, I = _paulis_1q(dtype)

    scale = 0.5  # matches utils default for 1q gates
    P = {"x": X, "y": Y, "z": Z}[axis]

    # Build a fixed reference unitary A
    alpha, beta, gamma = 0.37, -0.91, 1.13
    A = (
        _rot_from_generator(alpha, X, scale, dtype)
        @ _rot_from_generator(beta, Y, scale, dtype)
        @ _rot_from_generator(gamma, Z, scale, dtype)
    )

    # Parameter value
    theta0 = 0.42

    # Overlap z(theta) = Tr(A^† U(theta))
    def z(theta: float) -> jnp.ndarray:
        U = _rot_from_generator(theta, P, scale, dtype)
        return jnp.trace(jnp.conjugate(A).T @ U)

    z0 = z(theta0)
    dU = _dU_dtheta(theta0, P, scale, dtype)
    dz_dG = jnp.conjugate(A)  # ∂z/∂U = A^†

    n_sites = 1
    d = 2 ** n_sites
    denom = (d if is_normalized else d * d)
    analytic = -(2.0 / denom) * jnp.real(jnp.conjugate(z0) * jnp.trace(dz_dG.T @ dU))

    # Pick the mapped gradient function
    fn_map = {
        "rx": param_grad_rx,
        "ry": param_grad_ry,
        "rz": param_grad_rz,
    }
    param_fn = fn_map[fn_name]

    # Call implementation under test
    out = param_fn(
        theta=jnp.array([theta0]),
        dL_dG=dz_dG,
        L=z0,
        meta={"exp_scale": scale},
        n_sites=n_sites,
        is_normalized=is_normalized,
    )

    np.testing.assert_allclose(
        np.asarray(out[0]),
        np.asarray(analytic),
        rtol=1e-7,
        atol=1e-9,
        err_msg=f"param_grad_{fn_name} mismatch vs analytic"
    )


@pytest.mark.parametrize("is_normalized", [False, True])
def test_param_grad_xxyyzz_matches_analytic(is_normalized):
    # Setup: 2-qubit reference and exp(-i s (a XX + b YY + c ZZ)) gate
    dtype = jnp.complex128
    XX, YY, ZZ, I4 = _paulis_2q(dtype)

    scale = 1.0  # matches utils default for 2q xxyyzz gate

    # Reference 2-qubit unitary A: product of local rotations
    X, Y, Z, I2 = _paulis_1q(dtype)
    a1, b1, c1 = 0.21, -0.33, 0.47
    a2, b2, c2 = -0.19, 0.51, -0.73
    A_local_1 = (
        _rot_from_generator(a1, X, 0.5, dtype)
        @ _rot_from_generator(b1, Y, 0.5, dtype)
        @ _rot_from_generator(c1, Z, 0.5, dtype)
    )
    A_local_2 = (
        _rot_from_generator(a2, X, 0.5, dtype)
        @ _rot_from_generator(b2, Y, 0.5, dtype)
        @ _rot_from_generator(c2, Z, 0.5, dtype)
    )
    A = jnp.kron(A_local_1, A_local_2)

    # Parameters (a, b, c)
    theta = jnp.array([0.12, -0.31, 0.44])

    # Build U = exp(-i s a XX) exp(-i s b YY) exp(-i s c ZZ)
    Ua = _rot_from_generator(theta[0], XX, scale, dtype)
    Ub = _rot_from_generator(theta[1], YY, scale, dtype)
    Uc = _rot_from_generator(theta[2], ZZ, scale, dtype)
    U = Ua @ Ub @ Uc

    # Overlap and its Euclidean gradient
    z0 = jnp.trace(jnp.conjugate(A).T @ U)
    dz_dG = jnp.conjugate(A)

    # dU/da, dU/db, dU/dc
    dU_da = (-1j * scale) * (XX @ U)
    dU_db = (-1j * scale) * (YY @ U)
    dU_dc = (-1j * scale) * (ZZ @ U)

    n_sites = 2
    d = 2 ** n_sites
    denom = (d if is_normalized else d * d)
    ga = -(2.0 / denom) * jnp.real(jnp.conjugate(z0) * jnp.trace(dz_dG.T @ dU_da))
    gb = -(2.0 / denom) * jnp.real(jnp.conjugate(z0) * jnp.trace(dz_dG.T @ dU_db))
    gc = -(2.0 / denom) * jnp.real(jnp.conjugate(z0) * jnp.trace(dz_dG.T @ dU_dc))
    analytic = jnp.stack([ga, gb, gc])

    out = param_grad_xxyyzz(
        theta=theta,
        dL_dG=dz_dG,
        L=z0,
        meta={"exp_scale": scale},
        n_sites=n_sites,
        is_normalized=is_normalized,
    )

    np.testing.assert_allclose(
        np.asarray(out),
        np.asarray(analytic),
        rtol=1e-7,
        atol=1e-9,
        err_msg="param_grad_xxyyzz mismatch vs analytic"
    )

