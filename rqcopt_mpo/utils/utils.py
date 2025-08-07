import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer
from rqcopt_mpo.mpo.mpo_dataclass import MPO
import jax.numpy as jnp
from typing import Dict, Optional


def gate_map(layer: GateLayer, n_sites: int):
    """
    Generate mappings from qubit site indices to gates for a given GateLayer.

    This function scans through all gates in the provided layer and constructs two
    dictionaries:
    1. gate_map_left: Maps each starting site index to the Gate that begins there.
    2. gate_map_right: Maps each ending site index to the Gate that ends there.

    If multiple gates share the same starting or ending site, a warning is printed,
    and the last encountered gate at that site will be used in the mapping.

    Parameters:
        layer (GateLayer): The layer of gates containing Gate objects with qubit indices.
        n_sites (int): The total number of qubit sites to consider (indices 0 to n_sites-1).

    Returns:
        Tuple[Dict[int, Optional[Gate]], Dict[int, Optional[Gate]]]:
            - gate_map_left: A dictionary mapping starting site indices to their corresponding Gate or None.
            - gate_map_right: A dictionary mapping ending site indices to their corresponding Gate or None.
    """
    # Map starting site to gate
    gate_map_left: Dict[int, Optional[Gate]] = {i: None for i in range(n_sites)}
    # Map ending site to gate
    gate_map_right: Dict[int, Optional[Gate]] = {i: None for i in range(n_sites)}

    for gate in layer.gates:
        leftmost_site = min(gate.qubits)
        rightmost_site = max(gate.qubits)
        if leftmost_site < n_sites:
            if gate_map_left[leftmost_site] is not None:
                 print(f"Warning: Multiple gates starting at site {leftmost_site}. Using last found: {gate.name}.")
            gate_map_left[leftmost_site] = gate
        if rightmost_site < n_sites:
             if gate_map_right[rightmost_site] is not None:
                 print(f"Warning: Multiple gates ending at site {rightmost_site}. Using last found: {gate.name}.")
             gate_map_right[rightmost_site] = gate
    return gate_map_left, gate_map_right


def layer_to_matrix(layer: GateLayer, n_sites: int) -> jnp.ndarray:
    layer_to_matrix = 1.
    i = 0
    gate_map_left, _ = gate_map(layer=layer, n_sites=n_sites)
    while i < n_sites:
        gate = gate_map_left.get(i)

        if gate is None:
            layer_to_matrix = jnp.kron(layer_to_matrix, jnp.eye(2))
            i += 1
        elif gate.is_two_qubit():
            layer_to_matrix = jnp.kron(layer_to_matrix, gate.matrix)
            i += 2
        else:
            layer_to_matrix = jnp.kron(layer_to_matrix, gate.matrix)
            i += 1
    return layer_to_matrix

def compute_init_horizontal_dir(mpo_ref_adj: MPO, num_layers: int) -> str:
    """
    Computes the initial horizontal sweep direction based on the
    canonical form of a reference MPO and the number of layers.

    The logic is:
    - If num_layers is even, the initial direction should be opposite
      to the reference MPO's canonical direction.
    - If num_layers is odd, the initial direction should be the same
      as the reference MPO's canonical direction.

    Args:
        mpo_ref_adj (MPO): The reference MPO whose canonical form dictates the base direction.
                           It should have `is_left_canonical` and `is_right_canonical` attributes.
        num_layers (int): The number of layers involved.

    Returns:
        str: The computed initial direction, either "left_to_right" or "right_to_left".

    Raises:
        ValueError: If the reference MPO is not in a definite canonical form
                    (neither or both flags are set).
        ValueError: If num_layers is not a positive integer.
    """
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("num_layers must be a positive integer.")

    # Determine the base direction from the reference MPO's canonical form
    if mpo_ref_adj.is_left_canonical and not mpo_ref_adj.is_right_canonical:
        base_direction = "left_to_right"
    elif mpo_ref_adj.is_right_canonical and not mpo_ref_adj.is_left_canonical:
        base_direction = "right_to_left"
    else:
        # Handles cases where is_left_canonical == is_right_canonical (both False or both True)
        raise ValueError(
            "Reference MPO (mpo_ref_adj) must be in a definite left- or right-canonical form. "
            f"Got is_left_canonical={mpo_ref_adj.is_left_canonical}, "
            f"is_right_canonical={mpo_ref_adj.is_right_canonical}."
        )

    # Determine the initial direction based on layer parity
    if num_layers % 2 == 0:  # Even number of layers
        # Initial direction is opposite to the base direction
        if base_direction == "left_to_right":
            init_horizontal_dir = "right_to_left"
        else:  # base_direction == "right_to_left"
            init_horizontal_dir = "left_to_right"
    else:  # Odd number of layers
        # Initial direction is the same as the base direction
        init_horizontal_dir = base_direction

    return init_horizontal_dir

