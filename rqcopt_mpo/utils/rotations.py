import rqcopt_mpo.jax_config
import jax.numpy as jnp


# z := Trace(U_ref^adj U_circ), i.e. complex holomorphic function (i.e. dz/dG*=0)
# note: the loss is real: this also means (df/dz)* = df/dz*.

# Let G = G(θ) with real θ. Then
#
#     dL/dθ = ∑_i ∑_j (∂L/∂G_ij) (dG_ij/dθ) + ∑_i ∑_j (∂L/∂Ḡ_ij) (dḠ_ij/dθ).
#
# Since L is holomorphic, we have ∂L/∂Ḡ = 0. Therefore, the differential reduces to
#
#     dL/dθ = ∑_i ∑_j (∂L/∂G_ij) (dG_ij/dθ) = Tr(∂L/∂G^T dG/dθ), which is the sum implemented as
# 

def _backprop_hst_loss(z, dz_dG, dG_dtheta, n_sites, is_normalized):
    d = 2.0**n_sites
    denom = d*d if not is_normalized else d
    # implements  sum_{ij} (∂f/∂G_ij) * (dG_ij/dθ)
    return -(2.0 / denom) * jnp.real(jnp.conjugate(z) * jnp.trace(dz_dG.T @ dG_dtheta)
)  

def _rot_from_generator(theta, P, scale, dtype):
    # exp(-i * scale * theta * P) for an involutory P (P^2 = I)
    c = jnp.cos(scale * theta)
    s = jnp.sin(scale * theta)
    I = jnp.eye(P.shape[0], dtype=dtype)
    return c * I - 1j * s * P

def _d_rot_from_generator(theta, P, scale, dtype):
    """Derivative of _rot_from_generator wrt theta."""
    c = jnp.cos(scale * theta)
    s = jnp.sin(scale * theta)
    I = jnp.eye(P.shape[0], dtype=dtype)
    return -scale * s * I - 1j * scale * c * P

# Pauli 1q
def _paulis_1q(dtype):
    X = jnp.array([[0, 1],
                   [1, 0]], dtype=dtype)
    Y = jnp.array([[0, -1j],
                   [1j, 0]], dtype=dtype)
    Z = jnp.array([[1, 0],
                   [0, -1]], dtype=dtype)
    I = jnp.eye(2, dtype=dtype)
    return X, Y, Z, I

# Pauli 2q (tensor products)
def _paulis_2q(dtype):
    X, Y, Z, I2 = _paulis_1q(dtype)
    kron = jnp.kron
    XX = kron(X, X)
    YY = kron(Y, Y)
    ZZ = kron(Z, Z)
    I4 = jnp.eye(4, dtype=dtype)
    return XX, YY, ZZ, I4

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
