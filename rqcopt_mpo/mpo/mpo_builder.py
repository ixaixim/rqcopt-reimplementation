import jax.numpy as jnp
import jax
import numpy as np
from typing import List

from .mpo_dataclass import MPO

def get_id_mpo(nsites: int, dtype=jnp.complex64) -> MPO:
    """
    Constructs an MPO representing the identity operator on `nsites` qubits.

    Each local tensor is the 2x2 identity matrix with trivial virtual
    bond dimension 1. The tensor convention is (left_bond, phys_out, phys_in, right_bond).

    Args:
        nsites: The number of qubits (sites) for the identity MPO.
        dtype: The data type for the tensors (e.g., jnp.complex128).

    Returns:
        An MPO object representing the identity.
    """
    if nsites <= 0:
        raise ValueError("Number of sites must be positive.")

    # Local tensor: 2x2 identity reshaped to (1, 2, 2, 1)
    # Corresponds to (left_bond=1, phys_out=2, phys_in=2, right_bond=1)
    id_phys = jnp.eye(2, dtype=dtype)
    id_tensor = id_phys.reshape((1, 2, 2, 1))

    # Create the list of tensors
    identity_tensors = [id_tensor] * nsites
    # Note: Since JAX arrays are immutable, simple multiplication is fine.
    # If using mutable arrays (like standard numpy), use a list comprehension
    # with .copy(): [id_tensor.copy() for _ in range(nsites)]

    # Create and return the MPO object
    identity_mpo = MPO(tensors=identity_tensors)

    # The MPO __post_init__ will handle setting n_sites, dims etc.
    # It also checks bond consistency, which is 1->1 here, so it's valid.

    return identity_mpo


# ---------------------------------------------------------------------
# helper to draw a random complex tensor 
def _rand_complex(shape, *, dtype=jnp.complex64, key=None):
    if key is None:  # new random seed every time
        key = jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    k_re, k_im = jax.random.split(key)
    return (jax.random.normal(k_re, shape, dtype=jnp.float32) +
            1j * jax.random.normal(k_im, shape, dtype=jnp.float32)
           ).astype(dtype)
# ---------------------------------------------------------------------
def create_dummy_mpo(bond_dims_right, phys_dim=2, *, dtype=jnp.complex64,
                     random=False, seed=None):
    """
    Build a minimal :class:`MPO` whose tensors have the correct shapes,
    but contain no physical information.

    Parameters
    ----------
    bond_dims_right : Sequence[int]
        Right virtual bond dimension for every site (its length sets n_sites).
        The left bond of site *s* is simply bond_dims_right[s‑1] (with 1 for s=0).
        Use leading/trailing 1s for open boundaries.
    phys_dim : int, default 2
        Local physical Hilbert‑space dimension.
    dtype : jax.numpy dtype, default complex64
        Numerical dtype for all tensors.
    random : bool, default False
        • False → fill tensors with zeros (fast, sufficient for shape checks)  
        • True  → fill tensors with i.i.d. complex N(0,1) numbers
    seed : int | None
        Fix the random seed for reproducibility (ignored if random=False).

    Returns
    -------
    MPO
        A shape‑correct MPO instance ready for use in tests.
    """
    n_sites   = len(bond_dims_right)
    tensors   = []
    left_dim  = 1                           # open left boundary
    key_pool  = (jax.random.split(jax.random.PRNGKey(seed), n_sites)
                 if random and seed is not None else None)

    for s, right_dim in enumerate(bond_dims_right):
        shape = (left_dim, phys_dim, phys_dim, right_dim)
        if random:
            tensor = _rand_complex(shape, dtype=dtype,
                                   key=None if key_pool is None else key_pool[s])
        else:
            tensor = jnp.zeros(shape, dtype=dtype)

        tensors.append(tensor)
        left_dim = right_dim                # becomes next site's left bond

    return MPO(tensors)
