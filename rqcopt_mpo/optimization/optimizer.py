import rqcopt_mpo.jax_config

from typing import Optional, List, Dict
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, GateLayer, Gate 
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.mpo.mpo_builder import get_id_mpo
from rqcopt_mpo.tensor_network.core_ops import contract_mpo_with_layer_left_to_right, contract_mpo_with_layer, compress_SVD
from rqcopt_mpo.optimization.gradient import compute_layer_boundary_environments, _update_left_env, _update_right_env, gate_map, compute_gate_environment_tensor, compute_trace, compute_upper_lower_environments
import jax

# TODO: create a compute_full_bottom_environment function. to place in gradient.py, instead of doing it optimizer.py

# TODO: cleaner code would never let gate == None, if there is no gate at a site. Cleaner code would just use a single qubit identity gate. Then we can drop the None handling. 
#       This and lengthy syntax could be avoided if we had an iterator on the layer gates.
def optimize_circuit_local_svd(
    circuit_initial: Circuit,
    mpo_ref: MPO,
    num_sweeps: int,
    max_bondim_env: int,           # Max bond dim for environment MPOs
    svd_cutoff: float = 1e-12,     # Cutoff for gate update SVD
    layer_update_passes: int = 1,  # Passes within a layer (L<->R)
) -> Circuit:
    """
    Optimizes a quantum circuit using iterative local SVD updates based on
    environments computed against a reference MPO. Implements the sweep
    strategy described in Gibbs et al., arXiv:2409.16361v1.

    Args:
        circuit_initial: The initial Circuit object to optimize.
        mpo_ref: The reference MPO (e.g., V_Targ).
        num_sweeps: Number of full up-down sweeps through the circuit layers.
        max_bondim_env: Maximum bond dimension allowed when computing/updating
                        E_top and E_bottom environment MPOs.
        svd_cutoff: Singular value cutoff for the gate update SVD.
        layer_update_passes: Number of passes (e.g., L->R and R->L) within
                             each layer during a sweep.

    Returns:
        The optimized Circuit object.
    """
    # --- Initialization ---
    # 1. Copy the initial circuit to work on
    circuit = circuit_initial.copy() # Assuming Circuit has a deep copy method
    
    # 2. Get adjoint
    n_sites = mpo_ref.n_sites
    dtype = mpo_ref[0].dtype
    mpo_ref = mpo_ref.dagger()
    
    # 3. Get identity MPO
    mpo_identity = get_id_mpo(circuit.n_sites, dtype=dtype)

    # 4. Initialize environment caches 
    all_E_bottom: Dict[int, MPO] = {}

    # Loss tracking 
    loss_history = []




    # --- Main Sweep Loop ---
    for sweep in range(num_sweeps):
        print(f"\n--- Starting Sweep {sweep + 1}/{num_sweeps} ---")

        # Environment reset to solve error accumulation problems?

        # --- Step 1: Compute/Cache All Top Environments (Top-to-Bottom) ---
        # Initialization
        print("  Computing Top Environments...")
        top_layer_sweep_direction = 'left_to_right' if mpo_ref.is_right_canonical else 'right_to_left'
        all_E_top = compute_upper_lower_environments(
            mpo_ref=mpo_ref, circuit=circuit, direction='top', init_direction=top_layer_sweep_direction, max_bondim_env=max_bondim_env
            )

        # --- Step 2: Bottom-Up Sweep (Update Gates) ---
        print("  Bottom-Up Sweep...")
        E_bottom_current = mpo_identity 
        bottom_layer_sweep_direction = 'right_to_left' # init sweep direction for bottom env

        for l in range(circuit.num_layers): # 0 up to L-1
            print(f"    Layer {l}...")
            E_top_l = all_E_top[l] # Retrieve cached top environment
            current_layer = circuit.layers[l]
            # --- Inner loop(s) for updating gates within the layer ---
            for layer_update_pass in range(layer_update_passes):

                # --- Left-to-Right Pass within layer ---
                pass_losses_lr = _layer_pass_left_to_right(
                    current_layer, E_top_l, E_bottom_current, 
                    n_sites, dtype, max_bondim_env, svd_cutoff
                )
                loss_history.extend(pass_losses_lr)
                
                # --- Right-to-Left Pass within layer ---
                # Similar logic, compute left boundaries, sweep right-to-left
                # NOTE: the logic of the R-> L pass is different from the L->R, in the sense that it is more simple.
                pass_losses_lr = _layer_pass_right_to_left(
                    current_layer, E_top_l, E_bottom_current, 
                    n_sites, dtype, max_bondim_env, svd_cutoff
                )
                loss_history.extend(pass_losses_lr)
                
                

            # --- Update Bottom Environment for the *next* layer ---
                        
            # Only if not the last layer
            if l < circuit.num_layers - 1:
                E_bottom_current = contract_mpo_with_layer( # Choose L->R or R->L
                    E_bottom_current,
                    current_layer, # Use the *updated* layer
                    layer_is_below=False, # Contracting layer above E_bottom
                    max_bondim=max_bondim_env,
                    direction= bottom_layer_sweep_direction
                    # Pass sweep_direction if needed
                )
                bottom_layer_sweep_direction = 'right_to_left' if bottom_layer_sweep_direction == 'left_to_right' else 'left_to_right'
            
                # we store the bottom layers for later top-to-bottom sweep. 
                all_E_bottom[l+1] = E_bottom_current

        # --- Step 3: Top-Down Sweep (Mirror of Step 2) ---
        # NOTE: probably have to eliminate used stored environments.
        print("     \nTop-Down Sweep...")
        E_top_current = mpo_ref 
        top_layer_sweep_direction = 'left_to_right' if mpo_ref.is_right_canonical else 'right_to_left'

        # NOTE: if you plan to compute the trace, it might be better to store E_bottom by using compute_upper_lower_environments instead of caching. Reason: you might end up with mpo_ref and mpo version of circuit with canonicity in the same direction.
        #    Retrieve/Compute E_bottom_l: done at end of Bottom-up sweep 
        all_E_bottom[0] = mpo_identity
        for l in reversed(range(circuit.num_layers)): # L-1 up to zero
            print(f"    Layer {l}...")
            E_bottom_l = all_E_bottom[l] # Retrieve cached top environment
            current_layer = circuit.layers[l]

            # --- Inner loop(s) for updating gates within the layer ---
            for _ in range(layer_update_passes):

                # --- Left-to-Right Pass within layer ---
                pass_losses_lr = _layer_pass_left_to_right(
                    current_layer, E_top_current, E_bottom_l, 
                    n_sites, dtype, max_bondim_env, svd_cutoff
                )
                loss_history.extend(pass_losses_lr)
                
                # --- Right-to-Left Pass within layer ---
                # Similar logic, compute left boundaries, sweep right-to-left
                # NOTE: the logic of the R-> L pass is different from the L->R, in the sense that it is more simple.
                pass_losses_lr = _layer_pass_right_to_left(
                    current_layer, E_top_current, E_bottom_l, 
                    n_sites, dtype, max_bondim_env, svd_cutoff
                )
                loss_history.extend(pass_losses_lr)

            # --- Update Top Environment for the *next* layer ---
            
            # Only if not the last layer
            if l > 0:
                E_top_current = contract_mpo_with_layer( # Choose L->R or R->L
                    E_top_current,
                    current_layer, # Use the *updated* layer
                    layer_is_below=True, # Contracting layer above E_bottom
                    max_bondim=max_bondim_env,
                    direction= top_layer_sweep_direction
                    # Pass sweep_direction if needed
                )
                top_layer_sweep_direction = 'right_to_left' if top_layer_sweep_direction == 'left_to_right' else 'left_to_right'
            
                # we store the top layers for later bottom-to-top sweep. 
                all_E_top[l-1] = E_top_current
            # NOTE: to compute full trace at the end of sweep, you can further contract E_top_current when l==0.
                    # this will return the fully contracted circuit-MPO as an MPO, of which you have to compute the trace.




        # --- End of Sweep ---

    # return circuit with updated gates, return loss function. 
    return circuit, loss_history


def _layer_pass_left_to_right(
    current_layer: GateLayer,  # Modified in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
    max_bondim_env: int, # For boundary env computations if they truncate # TODO (boundary env computations do not have truncation atm)
    svd_cutoff: float
) -> List[float]:
    """
    Performs a left-to-right sweep within a single layer, updating gates.
    Modifies current_layer.gates in-place.
    """
    pass_loss_history = []
    print("      L->R Pass...")

    # 1. Compute all right boundary envs for this layer
    E_right_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side='right'
    )
    E_left_boundaries = [None] * n_sites
    # 2. Initialize E_left_current (e.g., identity tensor)
    E_left_current = jnp.eye(1, dtype=dtype)
    # 3. Loop through qubits in layer:

    gate_to_idx_map = {id(g): idx for idx, g in enumerate(current_layer.gates)}
    i = 0

    gate_map_left, _ = gate_map(current_layer, n_sites)
    # cache left env only if a gate starts at the site
    if gate_map_left.get(0) is not None:
        E_left_boundaries[0] = E_left_current # Environment left of site 0
    stop_index = max(s for s, g in gate_map_left.items() if g is not None)
    # NOTE: in the future, it would be better to have an iterator on the layer (from the left, from the right). It makes it easier to go through the sites.
    while i <= stop_index:
        # if gate starting at i is present, we check if it is single or two-qubit, or None
        gate = gate_map_left.get(i)

        if gate is None:
            # Contract the identity through the boundary MPOs
            E_left_current, _ = _update_left_env(
                E_left_current,
                E_top_l[i], E_bottom_current[i]
            )
            i += 1  
            continue               

        elif gate.is_two_qubit():

            #    a. Retrieve E_right_i from E_right_boundaries
            right_site_index = max(gate.qubits)
            E_right_current = E_right_boundaries[right_site_index]
            if E_right_current is None:
                print(f"Error: Missing right environment for gate ending at site {right_site_index}. Skipping update.")
            # Update left environment assuming no gate interaction for this step

            #    b. Env = compute_gate_environment_tensor(g_i.qubits, E_top_l, E_bottom_current, E_left_current, E_right_i) #TODO: in gradient.py
            
            Env = compute_gate_environment_tensor(gate.qubits, E_top_l, E_bottom_current, E_left_current, E_right_current)
            trace = compute_trace(Env, gate.tensor)
            pass_loss_history.append(trace)

            #    c. U, S, Vh = jnp.linalg.svd(Env, full_matrices=False)
            # Env to matrix: group tensor indices into matrix indexes, separating upper from lower indices. 
            out_dims = Env.shape[Env.ndim//2:]
            in_dims = Env.shape[:Env.ndim//2]
            matrix_env = Env.reshape(np.prod(out_dims), np.prod(in_dims))
            U, S, Vh = jnp.linalg.svd(matrix_env, full_matrices=False)
            if svd_cutoff is not None:
                U, S, Vh, k_trunc = compress_SVD(U, S, Vh, cutoff=svd_cutoff)
            #    d. Apply cutoff to S if needed
            
            new_gate_matrix = U @ Vh 
            new_gate = gate.copy()
            new_gate.matrix = new_gate_matrix

            # Store the updated tensor temporarily
            # updated_gate_tensors[gate_obj] = new_gate_tensor

            #    f. Update E_left_current by contracting with the *updated* g_i, E_top_l sites, E_bottom_current sites
            E_left_current, _ = _update_left_env(E_left_current, E_top_l[i], E_bottom_current[i], new_gate, E_top_l[i+1], E_bottom_current[i+1])
            
            original_gate_id = id(gate)
            idx_in_layer_list = gate_to_idx_map[original_gate_id]
            current_layer.gates[idx_in_layer_list] = new_gate
            i +=2  
            # loss_history.append(trace)
            continue

        elif gate.is_single_qubit():
            
            E_right_current = E_right_boundaries[i]
            if E_right_current is None:
                print(f"Error: Missing right environment for gate ending at site {i}. Skipping update.")
            Env = compute_gate_environment_tensor(gate.qubits, E_top_l, E_bottom_current, E_left_current, E_right_current)
            trace = compute_trace(Env, gate.tensor)
                # SVD update
            # TODO: there might be 5-10x faster method for single gate qubit update: use polar projection to minimize distance of Tr(E^dag G).
            # for 2x2 matrices there might be closed form O(1). for d>2 dxd matrices, this does not work, and it O(d^3), just like SVD. 
            U, S, Vh = jnp.linalg.svd(Env, full_matrices=False)
            if svd_cutoff is not None:
                U, S, Vh, k_trunc = compress_SVD(U, S, Vh, cutoff=svd_cutoff)
            new_gate_matrix = U @ Vh          # shape: (d_out, d_in)
            new_gate = gate.copy()
            new_gate.matrix = new_gate_matrix

            # Propagate left boundary one site to the right
            E_left_current, _ = _update_left_env(
                E_left_current,                 # rank‑k
                E_top_l[i], E_bottom_current[i],
                new_gate                 # the updated gate
            )

            original_gate_id = id(gate)
            idx_in_layer_list = gate_to_idx_map[original_gate_id]
            current_layer.gates[idx_in_layer_list] = new_gate
            i += 1
            continue

        else: 
            raise TypeError(
                f"Unsupported gate object at layer {current_layer.layer_index}, site{i}: "
                f"{gate!r} (expected single-qubit, two-qubit, or None)."
            )
    return pass_loss_history

def _layer_pass_right_to_left(
    current_layer: GateLayer,  # Modified in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
    max_bondim_env: int, # For boundary env computations if they truncate # TODO (boundary env computations do not have truncation atm)
    svd_cutoff: float
) -> List[float]:

    print("      R->L Pass...")
    pass_loss_history = []

    E_left_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side='left'
    )
    gate_idx_in_layer = 0
    E_right_current = jnp.eye(1, dtype=dtype) # Starts as env to the far right (virtual bond dim 1)
    _, gate_map_right = gate_map(current_layer, n_sites)

    # #    This is because SVD update creates a *new* Gate object, which then replaces the old one in the list.
    # gate_to_idx_map = {id(g): idx for idx, g in enumerate(current_layer.gates)}
    site_k = n_sites - 1 # Start from the rightmost MPO site index
    # NOTE: no need to go to zero, could end at first gate.      
    last_gate_at_idx = min(s for s, g in gate_map_right.items() if g is not None)

    while site_k >= last_gate_at_idx:
        gate = gate_map_right.get(site_k) 
        
        if gate is None: # No gate ending at site_k; treat as identity on site_k.
            E_right_current, step = _update_right_env(
                E_right_current,
                E_top_l[site_k], E_bottom_current[site_k],
            )
            site_k += step # Move to the next site to the left
            continue

        else:
            # A gate `gate` exists, and its rightmost qubit is `site_k`.
            qubits = gate.qubits
            leftmost_qb_of_gate = min(qubits)
            E_left_current = E_left_boundaries[leftmost_qb_of_gate] # Use .get for safety
            if E_left_current is None:
                print(f"Error: Missing right environment for gate ending at site {max(gate.qubits)}. Skipping update.")

            Env = compute_gate_environment_tensor(
                qubits, E_top_l, E_bottom_current, E_left_current, E_right_current
            )
            trace = compute_trace(Env, gate.tensor)
            pass_loss_history.append(trace)
            # Env tensor dimensions are assumed: (in_0, in_1, ..., out_0, out_1, ...)
            env_ndim = Env.ndim
            out_indices_shape = Env.shape[env_ndim//2:] # Shape of output physical legs (for the 'ket' part of the gate matrix)
            in_indices_shape = Env.shape[:env_ndim//2] # Shape of input physical legs (for the 'bra' part of the gate matrix)

            
            matrix_env = Env.reshape(np.prod(out_indices_shape), np.prod(in_indices_shape))
            U_svd, S_svd, Vh_svd = jnp.linalg.svd(matrix_env, full_matrices=False)
            if svd_cutoff is not None: # This step might not be standard if goal is strictly unitary from SVD
                U_svd, S_svd, Vh_svd, k_trunc = compress_SVD(U_svd, S_svd, Vh_svd, cutoff=svd_cutoff)
            
            new_gate_matrix = U_svd @ Vh_svd # Shape (TotalOutDim, TotalInDim)
            new_gate_obj = gate.copy() # Copies structure (qubits, name, etc.)
            new_gate_obj.matrix = new_gate_matrix 

            current_layer.gates[gate_idx_in_layer] = new_gate_obj

            # Update the right Environment
            if gate.is_two_qubit():
                E_right_current, step = _update_right_env(E_right_current, E_top_l[site_k], E_bottom_current[site_k], new_gate_obj, E_top_l[leftmost_qb_of_gate], E_bottom_current[leftmost_qb_of_gate])
                site_k += step
                continue
            elif gate.is_single_qubit():
                E_right_current, step = _update_right_env(E_right_current, E_top_l[site_k], E_bottom_current[site_k], new_gate_obj)
                site_k += step
                continue

    return pass_loss_history
