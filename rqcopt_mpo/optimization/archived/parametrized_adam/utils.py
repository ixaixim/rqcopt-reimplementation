from rqcopt_mpo.utils.rotations import _rot_from_generator, _paulis_1q, _paulis_2q, _backprop_hst_loss

import rqcopt_mpo.jax_config
import jax.numpy as jnp



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
    U(a,b,c) = exp(i * s * (a XX + b YY + c ZZ))
             = exp(i*s*a XX) exp(i*s*b YY) exp(i*s*c ZZ)  (since XX, YY, ZZ commute)
    ∂U/∂a = (i*s) XX U, and analogously for b, c.
    """
    if theta.shape[0] != 3:
        raise ValueError(f"Expected theta shape (3,), got {theta.shape}")
    scale = float(meta.get("exp_scale", 1.))
    dtype = dL_dG.dtype

    XX, YY, ZZ, _ = _paulis_2q(dtype)

    # Build U via commuting factors (fast closed forms).  Passing ``-P`` to
    # :func:`_rot_from_generator` produces ``exp(i * scale * theta * P)``.
    Ua = _rot_from_generator(theta[0], -XX, scale, dtype)
    Ub = _rot_from_generator(theta[1], -YY, scale, dtype)
    Uc = _rot_from_generator(theta[2], -ZZ, scale, dtype)
    U  = Ua @ Ub @ Uc

    dU_da = (1j * scale) * (XX @ U)
    dU_db = (1j * scale) * (YY @ U)
    dU_dc = (1j * scale) * (ZZ @ U)

    ga = _backprop_hst_loss(L, dL_dG, dU_da, n_sites, is_normalized)
    gb = _backprop_hst_loss(L, dL_dG, dU_db, n_sites, is_normalized)
    gc = _backprop_hst_loss(L, dL_dG, dU_dc, n_sites, is_normalized)
    return jnp.stack([ga, gb, gc])
