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
    h_left: float,
    h_right: float,
    coeff: float,
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
    IZ = two_qubit(I, Z)
    ZI = two_qubit(Z, I)
    H = H + h_left * ZI + h_right * IZ
    H = jnp.asarray(H, dtype=dtype)
    return jsp.expm(-1j * coeff * H)


def _local_evolution_xyz(
    *,
    Jx: float,
    Jy: float,
    Jz: float,
    hx_left: float,
    hy_left: float,
    hz_left: float,
    hx_right: float,
    hy_right: float,
    hz_right: float,
    coeff: float,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    """e^{-i · coeff · H_local} for XYZ model as a **jax.numpy** array (shape 4×4)."""
    X = pauli_x(dtype)
    Y = pauli_y(dtype)
    Z = pauli_z(dtype)
    I = identity(dtype)
    
    XX = two_qubit(X, X)
    YY = two_qubit(Y, Y)
    ZZ = two_qubit(Z, Z)

    H = Jx * XX + Jy * YY + Jz * ZZ
    IX = two_qubit(I, X)
    XI = two_qubit(X, I)
    IY = two_qubit(I, Y)
    YI = two_qubit(Y, I)
    IZ = two_qubit(I, Z)
    ZI = two_qubit(Z, I)
    
    H = H + hx_left * XI + hx_right * IX
    H = H + hy_left * YI + hy_right * IY
    H = H + hz_left * ZI + hz_right * IZ
        
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
        is_odd_layer = False
    else:  # "odd"
        bonds = range(1, n_sites - 1, 2)
        is_odd_layer = True

    # --- Populate the GateLayer -----------------------------------------------------
    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd_layer, n_sites=n_sites)

    for left in bonds:
        right = left + 1
        
        # Distribute field h: half for inner qubits, full for boundary
        h_left = h
        if left > 0:
            h_left /= 2.0
            
        h_right = h
        if right < n_sites - 1:
            h_right /= 2.0
            
        U_local = _local_evolution(
            J=J,
            D=D,
            h_left=h_left,
            h_right=h_right,
            coeff=coeff,
            dtype=dtype,
        )
        
        gate = Gate(
            matrix=U_local,
            qubits=(left, right),
            layer_index=layer_idx,
            name="exp(-iH)",
        )
        layer.add_gate(gate)

    return layer


def single_layer_trotterized_xyz(
    *,
    n_sites: int,
    Jx: float,
    Jy: float,
    Jz: float,
    hx: float,
    hy: float,
    hz: float,
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
        is_odd_layer = False
    else:  # "odd"
        bonds = range(1, n_sites - 1, 2)
        is_odd_layer = True

    # --- Populate the GateLayer -----------------------------------------------------
    layer = GateLayer(layer_index=layer_idx, is_odd=is_odd_layer, n_sites=n_sites)

    for left in bonds:
        right = left + 1
        
        # Distribute fields: half for inner qubits, full for boundary
        # Left qubit
        hxl, hyl, hzl = hx, hy, hz
        if left > 0:
            hxl /= 2.0; hyl /= 2.0; hzl /= 2.0
            
        # Right qubit
        hxr, hyr, hzr = hx, hy, hz
        if right < n_sites - 1:
            hxr /= 2.0; hyr /= 2.0; hzr /= 2.0
            
        U_local = _local_evolution_xyz(
            Jx=Jx, Jy=Jy, Jz=Jz,
            hx_left=hxl, hy_left=hyl, hz_left=hzl,
            hx_right=hxr, hy_right=hyr, hz_right=hzr,
            coeff=coeff,
            dtype=dtype,
        )

        gate = Gate(
            matrix=U_local,
            qubits=(left, right),
            layer_index=layer_idx,
            name="exp(-iH)",
        )
        layer.add_gate(gate)

    return layer

__all__ = [
    "single_layer_trotterized_heisenberg",
    "single_layer_trotterized_xyz",
]

