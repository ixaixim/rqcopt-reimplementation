import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer
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

