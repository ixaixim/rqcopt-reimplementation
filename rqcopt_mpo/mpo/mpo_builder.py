import jax.numpy as jnp
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
