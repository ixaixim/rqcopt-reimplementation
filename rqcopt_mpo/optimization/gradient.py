import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, List, Tuple
from rqcopt_mpo.tensor_network.core_ops import canonicalize_local_tensor, merge_two_mpos_and_gate, split_tensor_into_half_canonical_mpo_pair, contract_mpo_with_layer_right_to_left, contract_mpo_with_layer_left_to_right
from rqcopt_mpo.core_objects import Gate, GateLayer
from rqcopt_mpo.mpo.mpo_dataclass import MPO


# ---------------------------------------------------------------------
# 1.  SMALL, RE-USABLE HELPER
# ---------------------------------------------------------------------
def _update_left_env(
    L: jnp.ndarray,
    A1: jnp.ndarray,
    B1: jnp.ndarray,
    gate: Optional["Gate"] = None,
    A2: Optional[jnp.ndarray] = None,
    B2: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int]:
    """
    Contract the current *left* environment with one site (or two if a
    nearest-neighbour gate starts here).

    Returns
    -------
    new_L : jnp.ndarray   Updated environment
    step  : int           How many physical sites were consumed (1 or 2)
    """
    if gate is None:                       # no gate
        new_L = jnp.einsum(
            "ai, abcd, icbj -> dj", L, A1, B1, optimize="optimal"
        )
        return new_L, 1

    if gate.is_single_qubit():             # 1-qubit gate
        g = gate.tensor_4d
        new_L = jnp.einsum(
            "ai, abcd, ch, ihbj -> dj", L, A1, g, B1, optimize="optimal"
        )
        return new_L, 1

    if gate.is_two_qubit():                # 2-qubit gate
        if A2 is None or B2 is None:
            raise ValueError("Two-qubit gate requires A2/B2 tensors.")
        g = gate.tensor_4d
        new_L = jnp.einsum(
            "ai,abcd,defg,cfhk,ihbj,jkel->gl",
            L, A1, A2, g, B1, B2,
            optimize="optimal",
        )
        return new_L, 2

    raise ValueError("Unsupported gate type.")

# TODO: check that _update_left_env contractions are correct. 
# here are the contractions from the previous code for reference(proofed already)
            # if gate is not None and gate.is_two_qubit():

            #     # --- Two-Qubit Gate at (i, i+1) ---
            #     if i + 1 >= n_sites:
            #         print(f"Warning: Skipping 2Q gate {gate.name} at end of chain ({i},{i+1}).")
            #         # Fall through to treat site i as having no gate starting here
            #         gate = None # Effectively treat as no gate for update step
            #     else:
            #         A2 = E_top[i+1]
            #         B2 = E_bottom[i+1]
            #         # Ensure tensor_4d is valid
            #         gate_tensor = gate.tensor_4d
            #         if gate_tensor is None:
            #             print(f"Warning: Skipping 2Q gate {gate.name} due to invalid tensor_4d property.")
            #             # Reset L or use no-gate contraction? Resetting might be safer.
            #             # L = jnp.nan * L # Mark as invalid? Or default to no-gate contraction below?
            #             # i += 1 # Treat as single site without gate
            #             continue

            #         # Provided einsum for 2Q gate
            #         L = jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl', L, A1, A2, gate_tensor, B1, B2, optimize='optimal')
            #         i += 2 # Advance past the two sites

            #         # cache left env only if a gate starts at the site
            #         if gate_map_left.get(i) is not None:
            #             boundary_envs[i] = L # Store env left of site i+2

            #         continue # Skip single site update logic

            # # --- Single-Qubit Gate or No Gate at site i ---
            # if gate is not None and gate.is_single_qubit():
            #     # --- One-Qubit Gate ---
            #     gate_tensor = gate.tensor_4d # Should be (2,2)
            #     if gate_tensor.shape != (2, 2):
            #          print(f"Warning: Skipping 1Q gate {gate.name} due to non-2x2 shape {gate_tensor.shape}.")
            #          # Fall back to No Gate contraction
            #          L = jnp.einsum('ai, abcd, icbj -> dj', L, A1, B1, optimize='optimal')

            #     L = jnp.einsum('ai, abcd, ch, ihbj -> dj', L, A1, gate_tensor, B1, optimize='optimal') # Need to confirm B1 indices
            #     i += 1 # Advance one site

            # else:
            #     # --- No Gate ---
            #     L = jnp.einsum('ai, abcd, icbj -> dj', L, A1, B1, optimize='optimal')
            #     i += 1 # Advance one site

            # # cache left env only if a gate starts at the site
            # if gate_map_left.get(i) is not None:
            #     boundary_envs[i] = L # Store env left of site i+1



def _update_right_env(
    R: jnp.ndarray,
    A1: jnp.ndarray,
    B1: jnp.ndarray,
    gate: Optional["Gate"] = None,
    A0: Optional[jnp.ndarray] = None,
    B0: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, int]:
    """
    Contract the current *right* environment with one site (or two if a
    nearest-neighbour gate ends here).

    Returns
    -------
    new_R : jnp.ndarray   Updated environment
    step  : int           Negative number of sites to move left (-1 or -2)
    """
    if gate is None:                       # no gate
        new_R = jnp.einsum(
            "dj, abcd, icbj -> ai", R, A1, B1, optimize="optimal"
        )
        return new_R, -1

    if gate.is_single_qubit():             # 1-qubit gate
        g = gate.tensor_4d
        new_R = jnp.einsum(
            "dj, abcd, ch, ihbj -> ai", R, A1, g, B1, optimize="optimal"
        )
        return new_R, -1

    if gate.is_two_qubit():                # 2-qubit gate
        if A0 is None or B0 is None:
            raise ValueError("Two-qubit gate requires A0/B0 tensors.")
        g = gate.tensor_4d
        new_R = jnp.einsum(
            "gl,abcd,defg,cfhk,ihbj,jkel->ai",
            R, A0, A1, g, B0, B1,
            optimize="optimal",
        )
        return new_R, -2

    raise ValueError("Unsupported gate type.")

# TODO: check that the contractions from _update_right_env are correct. 
# here is the previous code for reference (proofed.)
            # if gate is not None and gate.is_two_qubit():
            #     # --- Two-Qubit Gate at (i-1, i) ---
            #     if i - 1 < 0:
            #          print(f"Error: 2Q gate {gate.name} ends at site {i} but goes out of bounds ({i-1}). Cannot contract.")
            #          break # Stop processing
            #     else:
            #         A0 = E_top[i-1]
            #         B0 = E_bottom[i-1]
            #         gate_tensor = gate.tensor_4d
            #         if gate_tensor is None:
            #              print(f"Warning: Skipping 2Q gate {gate.name} at ({i-1},{i}) due to invalid tensor_4d property.")
            #         else:
            #             # Contract with 2Q gate (derived path)
            #             R = jnp.einsum('gl,abcd,defg,cfhk,ihbj,jkel->ai', R, A0, A1, gate_tensor, B0, B1, optimize='optimal')
            #             next_i = i - 2 # Advance past the two sites

            # elif gate is not None and gate.is_single_qubit():
            #     # --- One-Qubit Gate ---
            #     gate_tensor = gate.tensor_4d
            #     if gate_tensor is None or gate_tensor.shape != (2, 2):
            #          print(f"Warning: Skipping 1Q gate {gate.name} at site {i} due to invalid shape {gate_tensor.shape if gate_tensor is not None else 'None'}.")
            #     else:
            #          R = jnp.einsum('dj, abcd, ch, ihbj -> ai', R, A1, gate_tensor, B1, optimize='optimal')
            #     next_i = i - 1

            # else:
            #     # --- No Gate ends at site i ---
            #     # Still need to contract the MPOs to propagate R
            #     R = jnp.einsum('dj, abcd, icbj -> ai', R, A1, B1, optimize='optimal')
            #     next_i = i - 1

def gate_map(layer: GateLayer, n_sites: int):
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



def compute_layer_boundary_environments(
    E_top: MPO,
    E_bottom: MPO,
    layer: GateLayer,
    side: str,
) -> List[jnp.ndarray]:
    """
    Computes all left or right boundary environment tensors for a given layer.

    These represent the contraction of everything to the left/right of each
    potential gate starting position within the layer, sandwiched between
    the top (E_top) and bottom (E_bottom) environment MPOs.

    Args:
        E_top: MPO representing the environment above the layer.
        E_bottom: MPO representing the environment below the layer.
        layer: The GateLayer object containing the gates.
        side: Which boundary environments to compute ('left' or 'right').

    Returns:
        A list of boundary environment tensors (or None). 
        The list will have length n_sites.
        If side='left', boundary_envs[i] is the environment to the LEFT of
        the gate STARTING at site i (or None if no gate starts at i).
        If side='right', boundary_envs[i] is the environment to the RIGHT of
        the gate ENDING at site i (or None if no gate ends at i).

    Raises:
        ValueError: If side is not 'left' or 'right', or if MPO sizes mismatch.
    """
    n_sites = E_top.n_sites
    if n_sites != E_bottom.n_sites:
        raise ValueError(f"E_top ({n_sites} sites) and E_bottom ({E_bottom.n_sites} sites) must have the same length.")
    if n_sites == 0:
        return []
    if side not in ['left', 'right']:
        raise ValueError("side must be 'left' or 'right'")

    dtype = E_top.tensors[0].dtype # Assume consistent dtypes

    # --- Prepare Gate Maps ---
    gate_map_left, gate_map_right = gate_map(layer, n_sites)

    # --- Compute Environments ---
    boundary_envs = [None] * (n_sites) # Env i is boundary to the left/right of site i

    if side == 'left':
        # --- Left Sweep ---
        # Initial L: contract virtual bonds of E_top[0] and E_bottom[0]
        # Assumes dummy bonds are dim 1. Needs shape (1, 1) or compatible.
        L = jnp.eye(1, dtype=dtype) # Assuming E_top/E_bottom start with bond dim 1
        if E_top[0].shape[0] != 1 or E_bottom[0].shape[0] != 1:
             print(f"Warning: Left boundary MPOs shapes ({E_top[0].shape}, {E_bottom[0].shape}) "
                   f"do not start with bond dim 1. Initial environment might be incorrect.")
             # Adjust L initialization if boundary bond dims are different
        
        # cache left env only if a gate starts at the site
        if gate_map_left.get(0) is not None:
                    boundary_envs[0] = L # Environment left of site 0

        i = 0 # sweeping index
        # stop left sweep after we computed the left environment of the rightmost gate in the layer
        stop_index = max(s for s, g in gate_map_left.items() if g is not None)

        while i < stop_index:
            
            A1 = E_top[i]       # Top MPO at site i
            B1 = E_bottom[i]    # Bottom MPO at site i
            gate = gate_map_left.get(i) # Gate starting at site i
            
            # grab next-site MPOs only when required
            A2 = B2 = None
            if gate is not None and gate.is_two_qubit():
                A2, B2 = E_top[i + 1], E_bottom[i + 1]
            L, step = _update_left_env(L, A1, B1, gate, A2, B2)
            i += step

            if i < n_sites and gate_map_left.get(i) is not None:
                boundary_envs[i] = L

    elif side == 'right':
        # --- Right Sweep ---
        R = jnp.eye(1, dtype=dtype) # Assuming E_top/E_bottom end with bond dim 1
        if E_top[-1].shape[-1] != 1 or E_bottom[-1].shape[-1] != 1:
             print(f"Warning: Right boundary MPOs shapes ({E_top[-1].shape}, {E_bottom[-1].shape}) "
                   f"do not end with bond dim 1. Initial environment R might be incorrect.")
             # Re-initialize R if needed

        # Store initial environment if a gate ends at site n_sites-1
        if gate_map_right.get(n_sites-1) is not None:
             boundary_envs[n_sites-1] = R

        # Determine the leftmost site a gate ends at
        start_index = min(s for s, g in gate_map_right.items() if g is not None)

        i = n_sites - 1
        while i > start_index : # Loop down to the first gate end or start of chain
            A1 = E_top[i]       # Top MPO at site i
            B1 = E_bottom[i]    # Bottom MPO at site i
            gate = gate_map_right.get(i) # Gate ending at site i

            A0 = B0 = None
            if gate is not None and gate.is_two_qubit() and i - 1 >= 0:
                A0, B0 = E_top[i - 1], E_bottom[i - 1]

            R, step = _update_right_env(R, A1, B1, gate, A0, B0)
            next_i = i + step        # step is negative


            # Store environment R if a gate ends at the *next* position (to the left)
            if next_i >= 0 and gate_map_right.get(next_i) is not None:
                boundary_envs[next_i] = R

            i = next_i # Move to the next position


    # --- Final Check ---
    if side == "left":
        n_gates = sum(g is not None for g in gate_map_left.values())
    else:                               # side == "right"
        n_gates = sum(g is not None for g in gate_map_right.values())

    n_envs = sum(env is not None for env in boundary_envs)

    if n_envs != n_gates:
        raise RuntimeError(
            f"compute_layer_boundary_environments: mismatch detected "
            f"({n_envs} boundary envs, {n_gates} gate(s) in layer)."
        )

    return boundary_envs



def compute_gate_environment_tensor(
    gate_qubits: Tuple[int, ...],
    E_top_layer: MPO,
    E_bottom_layer: MPO,
    E_left_boundary: jnp.ndarray,
    E_right_boundary: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the environment tensor for a specific gate location.

    Contracts the relevant site tensors from E_top and E_bottom with the
    provided left and right boundary tensors.

    Args:
        gate_qubits: The tuple of qubits the gate acts on (e.g., (i,) or (i, i+1)).
        E_top_layer: The top environment MPO for the current layer.
        E_bottom_layer: The bottom environment MPO for the current layer.
        E_left_boundary: The tensor representing the environment to the left.
        E_right_boundary: The tensor representing the environment to the right.

    Returns:
        The environment tensor, with indices corresponding to the gate's
        output and input physical legs. The index order
        should match the gate tensor convention (e.g., out1, out2, in1, in2).
    """
    n_sites = len(E_top_layer)
    if n_sites != len(E_bottom_layer):
        raise ValueError(f"E_top ({n_sites} sites) and E_bottom ({len(E_bottom_layer)} sites) must have the same length.")
    if n_sites == 0:
        raise ValueError("Cannot compute environment for MPOs with zero sites.")

    num_qubits = len(gate_qubits)
    # 1. Identify the site indices involved
    if num_qubits == 1:
        i = gate_qubits[0]
        # Check bounds
        if not (0 <= i < n_sites):
            raise ValueError(f"Single qubit index {i} out of bounds for MPO with {n_sites} sites.")
        print(f"  Calculating environment for 1Q gate at site {i}")
        indices = (i,)
        ip1 = None # Explicitly mark as None for clarity

        # 2. Extract the relevant MPO tensors
        E_top_tensors = (E_top_layer[i],)
        E_bottom_tensors = (E_bottom_layer[i],)
        environment_tensor = jnp.einsum('ab, acde, bfcg, eg -> fd', E_left_boundary, E_top_tensors[0], E_top_tensors[1], E_bottom_tensors[0], E_bottom_tensors[1], E_right_boundary)
        # TODO: take conjugate? transpose?

    elif num_qubits == 2:
        # Assume adjacent qubits for standard environment calculation
        i = min(gate_qubits)
        ip1 = max(gate_qubits)
        if ip1 != i + 1:
            # This calculation typically assumes adjacent sites for the direct contraction.
            # Non-adjacent gates require a different, more complex contraction scheme
            # (like iteratively contracting intermediate MPO sites into boundary tensors),
            # which is not implemented here.
            raise ValueError(f"Environment calculation currently assumes adjacent qubits for 2Q gates. Got {gate_qubits}.")
        # Check bounds
        if not (0 <= i < n_sites and 0 <= ip1 < n_sites):
             raise ValueError(f"Two qubit indices {(i, ip1)} out of bounds for MPO with {n_sites} sites.")
        print(f"  Calculating environment for 2Q gate at sites ({i}, {ip1})")
        indices = (i, ip1)

        # 2. Extract the relevant MPO tensors
        E_top_tensors = (E_top_layer[i], E_top_layer[ip1])
        E_bottom_tensors = (E_bottom_layer[i], E_bottom_layer[ip1])

        # we take the notation: for the environment (out1, out2, in1, in2), so that a Hilbert-Schmidt inner product L(X) = <E,X> = Tr(E^dag X), where E is the environment tensor obtained by "cutting out" X from the bra-ket network.
        # the trace is expressed as jnp.trace(environment_part.transpose(2,3,0,1).reshape((4,4)).conj() @ gate_tensor.reshape(4,4))
        environment_tensor = jnp.einsum('ab, acde, efgh, bick, kjfl, hl -> ijdg', E_left_boundary, E_top_tensors[0], E_top_tensors[1], E_bottom_tensors[0], E_bottom_tensors[1], E_right_boundary)
    else:
        raise ValueError(f"Unsupported number of gate qubits: {num_qubits}. Expected 1 or 2.")
    
    return environment_tensor

def compute_trace(Environment: jnp.ndarray, gate_tensor: jnp.ndarray):
    # computes the loss function from the Hilbert-Schmidt inner product L(X) = <E,X> = Tr(E^dag X), where E is the environment tensor obtained by "cutting out" X from the bra-ket network.
    # E has the (out1, out2, in1, in2) ordering.  
    trace = jnp.einsum('ijdg, ijdg ->', Environment.conj(), gate_tensor)
    # or equivalently:
    # trace_val = jnp.trace(environment_part.transpose(2,3,0,1).conj().reshape((4,4)) @ gate_tensor.reshape(4,4)) # equivalent, to use if environment is stored with (in1, in2, out1, out2)

    return trace
# def calculate_cost_function(circuit: Circuit, mpo_ref: MPO, max_bondim: int, **kwargs) -> float:

# TODO: cost function = derivative + plug in the gate