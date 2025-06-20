
import rqcopt_mpo.jax_config  # ensures JAX defaults are consistent

import jax.numpy as jnp
from . import operators  # Import I, X, Y, Z from the operators.py file


class HeisenbergModel:
    """
    Factory for representations of the 1D Heisenberg XXZ model Hamiltonian.

    H = J * Σᵢ(XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + Δ·ZᵢZᵢ₊₁) + h * ΣᵢZᵢ

    Attributes:
        n_sites (int): Number of sites (spins) in the chain.
        J (float): The overall coupling strength.
        Delta (float): The anisotropy parameter.
        h (float): The external magnetic field strength in the z-direction.
        dtype (jnp.dtype): The data type for the tensors (e.g., jnp.complex128).
    """
    def __init__(
        self,
        n_sites: int,
        J: float,
        Delta: float,
        *,
        h: float = 0.0,
        dtype: jnp.dtype = jnp.complex128
    ):
        if n_sites < 2:
            raise ValueError("Hamiltonian requires at least 2 sites.")
        self.n_sites = n_sites
        self.J = J
        self.Delta = Delta
        self.h = h
        self.d = 2  # Physical dimension
        self.dtype = dtype

    def __repr__(self) -> str:
        return (f"HeisenbergModel(n_sites={self.n_sites}, J={self.J}, "
                f"Delta={self.Delta}, h={self.h})")

    def _get_local_term_matrix(self, op: jnp.ndarray, site_idx: int) -> jnp.ndarray:
        """Pads a local operator to act on the full Hilbert space."""
        # Identity on the left
        H = jnp.eye(2**site_idx, dtype=self.dtype) if site_idx > 0 else 1
        # Operator on site i (and i+1 if two-site)
        H = jnp.kron(H, op)
        # Identity on the right
        num_remaining_sites = self.n_sites - site_idx - op.shape[0].bit_length() + 1 # op.shape[0].bit_length() is either 2 or 1 depending the op is one site or two site.
        H = jnp.kron(H, jnp.eye(2**num_remaining_sites, dtype=self.dtype)) if num_remaining_sites > 0 else H
        return H

    def build_hamiltonian_matrix(self) -> jnp.ndarray:
        """
        Construct the full, dense Hamiltonian matrix for the XXZ chain.

        Returns:
            jnp.ndarray: A (2**n_sites, 2**n_sites) matrix representing the Hamiltonian.
        """
        ham_matrix = jnp.zeros((2**self.n_sites, 2**self.n_sites), dtype=self.dtype)

        # Interaction terms
        op_XX = jnp.kron(operators.X, operators.X)
        op_YY = jnp.kron(operators.Y, operators.Y)
        op_ZZ = jnp.kron(operators.Z, operators.Z)

        for i in range(self.n_sites - 1):
            interaction_term = self.J * (op_XX + op_YY + self.Delta * op_ZZ)
            ham_matrix += self._get_local_term_matrix(interaction_term, i)

        # Magnetic field term
        if self.h != 0.0:
            for i in range(self.n_sites):
                field_term = self.h * operators.Z
                ham_matrix += self._get_local_term_matrix(field_term, i)

        return ham_matrix
