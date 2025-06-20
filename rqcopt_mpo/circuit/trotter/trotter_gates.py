import rqcopt_mpo.jax_config  # ensures JAX defaults are consistent

"""trotter_gates.py

Utility operators used by single_layer_trotterized_heisenberg and friends.

This file defines single-qubit Pauli matrices **X, Y, Z**, the identity **I**, and
all two-qubit tensor-products that show up in a nearest-neighbour Heisenberg
model ( **XX, YY, ZZ, IZ, ZI** ).  Everything is constructed with **JAX** and can
optionally be generated in a custom complex dtype (e.g. *jnp.complex64* for GPU
or *complex128* for extra precision).

The module exposes two layers of API:

1.  Stateless helper functions such as :pyfunc:`pauli_x` or
    :pyfunc:`two_qubit` that let you create fresh operators on demand.
"""

from typing import Sequence

import jax.numpy as jnp
import jax.scipy.linalg as jsp
from rqcopt_mpo.circuit.circuit_dataclasses import GateLayer, Gate

# -----------------------------------------------------------------------------
# Single-qubit operators
# -----------------------------------------------------------------------------

def identity(dtype: jnp.dtype) -> jnp.ndarray:  # I
    return jnp.eye(2, dtype=dtype)


def pauli_x(dtype: jnp.dtype) -> jnp.ndarray:  # X
    return jnp.array([[0, 1], [1, 0]], dtype=dtype)


def pauli_y(dtype: jnp.dtype) -> jnp.ndarray:  # Y
    return jnp.array([[0, -1j], [1j, 0]], dtype=dtype)


def pauli_z(dtype: jnp.dtype) -> jnp.ndarray:  # Z
    return jnp.array([[1, 0], [0, -1]], dtype=dtype)

# -----------------------------------------------------------------------------
# Two-qubit helpers
# -----------------------------------------------------------------------------

def two_qubit(op_left: jnp.ndarray, op_right: jnp.ndarray) -> jnp.ndarray:
    """Return *op_left ⊗ op_right* as a JAX array."""
    return jnp.kron(op_left, op_right)

# -----------------------------------------------------------------------------
# Evolution helpers
# -----------------------------------------------------------------------------

def _local_evolution(
    *,
    J: float,
    D: float,
    h: float,
    coeff: float,
    include_field: bool,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """e^{-i · coeff · H_local} as a **jax.numpy** array (shape 4×4)."""
    X = pauli_x(dtype)
    Y = pauli_y(dtype)
    Z = pauli_z(dtype)
    I = identity(dtype)
    XX = two_qubit(X, X)
    YY = two_qubit(Y, Y)
    ZZ = two_qubit(Z, Z)

    H = J * (XX + YY) + D * ZZ
    if include_field:
        IZ = two_qubit(I, Z)
        ZI = two_qubit(Z, I)
        H = H + h * (IZ + ZI)
    H = jnp.asarray(H, dtype=dtype)
    return jsp.expm(-1j * coeff * H)


def single_layer_trotterized_heisenberg(
    *,
    n_sites: int,
    J: float,
    D: float,
    h: float,
    coeff: float,
    parity: str,
    layer_idx: int,
    dtype: jnp.dtype,
) -> GateLayer:
    # --- Validation ----------------------------------------------------------------
    if n_sites % 2:
        raise ValueError(f"n_sites must be even, got {n_sites}")
    if parity not in ("odd", "even"):
        raise ValueError("parity must be 'odd' or 'even'")

    # --- Choose the bond list -------------------------------------------------------
    if parity == "even":
        bonds: Sequence[int] = range(0, n_sites - 1, 2)
        include_field = True
        is_odd_layer = False
    else:  # "odd"
        bonds = range(1, n_sites - 1, 2)
        include_field = False
        is_odd_layer = True

    # --- Pre-build the evolution matrix (shared by every gate in the layer) --------
    U_local = _local_evolution(
        J=J,
        D=D,
        h=h,
        coeff=coeff,
        include_field=include_field,
        dtype=dtype,
    )

    # --- Populate the GateLayer -----------------------------------------------------
    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd_layer, n_sites=n_sites)

    for left in bonds:
        gate = Gate(
            matrix=U_local,
            qubits=(left, left + 1),
            layer_index=layer_idx,
            name="exp(-iH)",
        )
        layer.add_gate(gate)

    return layer

__all__ = [
    "single_layer_trotterized_heisenberg",
]

