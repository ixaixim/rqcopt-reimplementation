import rqcopt_mpo.jax_config

from typing import List
import numpy as np
import jax.numpy as jnp

def global_loss(loss: List, n_sites: int, normalize: bool = False) -> float:
    if normalize:
        hst = 1 - 1/2**(n_sites) * np.abs(loss)**2
    else:
        hst = 1 - 1/2**(2*n_sites) * np.abs(loss)**2
    return hst

def overlap_to_loss(
    overlap: jnp.ndarray,
    *,
    kind: str = "HST",
    n_sites: int = None,
    normalize: bool = True
) -> jnp.ndarray:
    """
    Turn a raw sum-of-traces (overlap) into a scalar loss.
    
    Parameters
    ----------
    overlap
        The real (or complex) scalar overlap returned by the gradient routine.
    kind : {'HST','frobenius'}
        Which cost function you’re using:
        - 'HST':      Squared HST loss:
            1 - |Tr(U_ref^† U_circ)|^2 / (||U_ref||_F^2 * ||U_circ||_F^2)

        - 'frobenius': loss ∝ ||U_ref - U_circ||_F^2
    n_sites
        Number of physical sites in the chain (for HST normalization).
    normalize
        Whether the MPO reference is normalized. Note: the circuit is always unitary (not normalized).
    """
    if kind.lower() == "hst":
        # maximal overlap for d-dim local Hilbert spaces is d**n_sites
        if (normalize is False):
            # MPO
            denom = 2.0 ** (2 * n_sites)
        else: 
            denom = 2.0 ** n_sites
        return 1.0 - (jnp.abs(overlap) ** 2) / denom
    
    elif kind.lower() == "frobenius":
        if normalize:
            denom = 2.0 ** (n_sites / 2.0)
        else:
            denom = 2.0 ** n_sites
        return 2.0 - 2.0 * jnp.real(overlap) / denom
    else:
        raise ValueError(f"Unknown loss kind: {kind}")
