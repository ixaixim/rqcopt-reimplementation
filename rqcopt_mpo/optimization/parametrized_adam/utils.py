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
    d = 2**n_sites
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

# ---------- RX / RY / RZ parameter-grad mappers ----------
def param_grad_rx(theta: jnp.ndarray, dL_dG: jnp.ndarray, L, meta: dict, n_sites: int, is_normalized: bool) -> jnp.ndarray:
    # theta shape (1,)
    scale = float(meta.get("exp_scale", 0.5))
    dtype = dL_dG.dtype
    X, _, _, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], X, scale, dtype)        # 2x2
    dU_dtheta = (-1j * scale) * (X @ U)                        # (-i*s) X U
    g = _backprop_hst_loss(L, dL_dG, dU_dtheta, n_sites, is_normalized)
    return jnp.array([g])

def param_grad_ry(theta: jnp.ndarray, dL_dG: jnp.ndarray, L, meta: dict, n_sites: int, is_normalized: bool) -> jnp.ndarray:
    scale = float(meta.get("exp_scale", 0.5))
    dtype = dL_dG.dtype
    _, Y, _, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], Y, scale, dtype)
    dU_dtheta = (-1j * scale) * (Y @ U)
    g = _backprop_hst_loss(L, dL_dG, dU_dtheta, n_sites, is_normalized)
    return jnp.array([g])

def param_grad_rz(theta: jnp.ndarray, dL_dG: jnp.ndarray, L, meta: dict, n_sites: int, is_normalized: bool) -> jnp.ndarray:
    scale = float(meta.get("exp_scale", 0.5))
    dtype = dL_dG.dtype
    _, _, Z, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], Z, scale, dtype)
    dU_dtheta = (-1j * scale) * (Z @ U)
    g = _backprop_hst_loss(L, dL_dG, dU_dtheta, n_sites, is_normalized)
    return jnp.array([g])

# ---------- R = a XX + b YY + c ZZ parameter-grad mapper ----------
def param_grad_xxyyzz(theta: jnp.ndarray, dL_dG: jnp.ndarray, L, meta: dict, n_sites: int, is_normalized: bool) -> jnp.ndarray:
    """
    theta = (a, b, c)
    U(a,b,c) = exp(-i * s * (a XX + b YY + c ZZ))
             = exp(-i*s*a XX) exp(-i*s*b YY) exp(-i*s*c ZZ)  (since XX, YY, ZZ commute)
    ∂U/∂a = (-i*s) XX U, and analogously for b, c.
    """
    if theta.shape[0] != 3:
        raise ValueError(f"Expected theta shape (3,), got {theta.shape}")
    scale = float(meta.get("exp_scale", 1.))
    dtype = dL_dG.dtype

    XX, YY, ZZ, I4 = _paulis_2q(dtype)

    # Build U via commuting factors (fast closed forms)
    Ua = _rot_from_generator(theta[0], XX, scale, dtype)
    Ub = _rot_from_generator(theta[1], YY, scale, dtype)
    Uc = _rot_from_generator(theta[2], ZZ, scale, dtype)
    U  = Ua @ Ub @ Uc

    dU_da = (-1j * scale) * (XX @ U)
    dU_db = (-1j * scale) * (YY @ U)
    dU_dc = (-1j * scale) * (ZZ @ U)

    ga = _backprop_hst_loss(L, dL_dG, dU_da, n_sites, is_normalized)
    gb = _backprop_hst_loss(L, dL_dG, dU_db, n_sites, is_normalized)
    gc = _backprop_hst_loss(L, dL_dG, dU_dc, n_sites, is_normalized)
    return jnp.stack([ga, gb, gc])
