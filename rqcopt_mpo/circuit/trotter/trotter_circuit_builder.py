import rqcopt_mpo.jax_config  # ensures JAX defaults are consistent

from typing import Tuple, Union, List

import jax.numpy as jnp

from rqcopt_mpo.circuit.circuit_dataclasses import GateLayer
from rqcopt_mpo.circuit.trotter.trotter_gates import single_layer_trotterized_heisenberg

def trotterized_heisenberg_layers(
    n_sites: int,
    J: float,
    D: float,
    h: float = 0.0,
    *,                       # everything after this is keyword-only
    order: int = 1,
    dt: float,
    reps: int,
    dtype: jnp.dtype | None = None,
) -> List[GateLayer]:
    """Return a list of :class:`GateLayer` objects implementing a Suzuki–Trotter
    approximation of *exp(-i·t·H)* for an **even-length** Heisenberg chain.

    Parameters
    ----------
    n_sites
        Total number of qubits (must be **even**).
    J / D / h
        Couplings in *H = J (X₁X₂ + Y₁Y₂) + D Z₁Z₂ + h (Z₁ + Z₂)*.
    order
        Trotter–Suzuki order (currently only ``1`` is implemented).
    dt
        Elementary time step **per bond layer**.
    reps
        How many times the elementary Trotter sequence is repeated.
    dtype
        Optional override for the JAX dtype of every gate matrix.

    Returns
    -------
    List[GateLayer]
        The brick-wall circuit layers in execution order (layer indices are
        already assigned consecutively starting from 0).
    """
    # --- Validation ---------------------------------------------------------
    if order not in (1, 2, 4):
        raise ValueError(f"order must be 1, 2 or 4 (got {order})")
    if n_sites % 2:
        raise ValueError(f"n_sites must be even (got {n_sites})")

    def _add_layer(parity: str, coeff: float, idx: int) -> GateLayer:
        return single_layer_trotterized_heisenberg(
            n_sites=n_sites,
            J=J,
            D=D,
            h=h,
            coeff=coeff,
            parity=parity,
            layer_idx=idx,
            dtype=dtype,
        )

    layers: List[GateLayer] = []
    layer_idx = 0

    # -----------------------------------------------------------------------
    # 1st-order Lie–Trotter  →  E(dt) O(dt)  repeated
    # -----------------------------------------------------------------------
    if order == 1:
        for _ in range(reps):
            for parity in ("even", "odd"):          # same coeff for both
                layers.append(_add_layer(parity, dt, layer_idx))
                layer_idx += 1
        return layers

    if order == 2: # 2*reps + 1 layers
        # --- leading E(dt/2) ----------------------------------------------
        layers.append(_add_layer("even", dt / 2, layer_idx)); layer_idx += 1

        # --- middle blocks  [O(dt) E(dt)]^{reps-1}  ------------------------
        for _ in range(reps - 1):
            layers.append(_add_layer("odd", dt, layer_idx));  layer_idx += 1
            layers.append(_add_layer("even", dt, layer_idx)); layer_idx += 1

        # --- trailing O(dt) E(dt/2) ---------------------------------------
        layers.append(_add_layer("odd",  dt,      layer_idx)); layer_idx += 1
        layers.append(_add_layer("even", dt / 2,  layer_idx)); layer_idx += 1

        return layers
    
    # ======================================================================
    # 4th-order  (Yoshida, minimal 6·reps + 1 layers)
    # ======================================================================
    # Yoshida constant  s = 1 / (2 − 2^{1/3})
    cbrt2 = 2.0 ** (1.0 / 3.0)
    s = 1.0 / (2.0 - cbrt2)

    # time-step coefficients
    a = s * dt                  #  a  = s·dt        (positive)
    b = (1.0 - 2.0 * s) * dt    #  b  = (1−2s)·dt   (negative)
    e_half = a / 2.0            # leading / trailing  E(s·dt/2)
    e_mid  = (1.0 - s) * dt / 2 # central even layers E((1−s)·dt/2)
    e_full = a                  # inter-rep merged   E(s·dt)


    # ---- leading  E(s·dt/2) ---------------------------------------------
    layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

    # ---- repeat block  ---------------------------------------------------
    for rep in range(1, reps + 1):
        #  O(a)
        layers.append(_add_layer("odd",  a,      layer_idx)); layer_idx += 1
        #  E((1−s)·dt/2)
        layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
        #  O(b)
        layers.append(_add_layer("odd",  b,      layer_idx)); layer_idx += 1
        #  E((1−s)·dt/2)
        layers.append(_add_layer("even", e_mid,  layer_idx)); layer_idx += 1
        #  O(a)
        layers.append(_add_layer("odd",  a,      layer_idx)); layer_idx += 1

        #  Inter-rep merger  E(s·dt)  (skip after final repetition)
        if rep < reps:
            layers.append(_add_layer("even", e_full, layer_idx)); layer_idx += 1

    # ---- trailing  E(s·dt/2) --------------------------------------------
    layers.append(_add_layer("even", e_half, layer_idx)); layer_idx += 1

    return layers

__all__ = [
    "trotterized_heisenberg_layers",
]
