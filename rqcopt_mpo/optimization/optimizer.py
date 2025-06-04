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
    mpo_ref = mpo_ref.dagger()
    init_direction = 'left_to_right' if mpo_ref.is_right_canonical else 'right_to_left'
    
    # 3. Get identity MPO
    mpo_identity = get_id_mpo(circuit.n_sites, dtype=mpo_ref.tensors[0].dtype)

    # 4. Initialize environment caches 
    all_E_bottom: Dict[int, MPO] = {}

    # Loss tracking 
    loss_history = []



    # --- Main Sweep Loop ---
    for sweep in range(num_sweeps):
        print(f"\n--- Starting Sweep {sweep + 1}/{num_sweeps} ---")

        # Environment reset to solve error accumulation problems?

        # --- Step 1: Compute/Cache All Top Environments (Top-to-Bottom) ---
        print("  Computing Top Environments...")
        all_E_top = compute_upper_lower_environments(mpo_ref=mpo_ref, circuit=circuit, direction='top', init_direction=init_direction, max_bondim_env=max_bondim_env)

        # --- Step 2: Bottom-Up Sweep (Update Gates) ---
        print("  Bottom-Up Sweep...")
        E_bottom_current = mpo_identity 
        for l in range(circuit.num_layers): # 0 up to L-1
            print(f"    Layer {l}...")
            E_top_l = all_E_top[l] # Retrieve cached top environment
            current_layer = circuit.layers[l]
            n_sites = E_top_l.n_sites
            dtype = E_top_l[0].dtype

            # --- Inner loop(s) for updating gates within the layer ---
            for _ in range(layer_update_passes):

                # --- Left-to-Right Pass within layer ---
                print("      L->R Pass...")
                # 1. Compute all right boundary envs for this layer
                E_right_boundaries = compute_layer_boundary_environments(
                    E_top_l, E_bottom_current, current_layer, side='right'
                )
                E_left_boundaries = [None] * n_sites
                # 2. Initialize E_left_current (e.g., identity tensor)
                E_left_current = jnp.eye(1, dtype=dtype)
                # 3. Loop through qubits in layer:

                i = 0
                gate_idx_in_layer = 0
                gate_map_left, gate_map_right = gate_map(current_layer, n_sites)
                # cache left env only if a gate starts at the site
                if gate_map_left.get(0) is not None:
                    E_left_boundaries[0] = E_left_current # Environment left of site 0
                stop_index = max(s for s, g in gate_map_left.items() if g is not None)

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

                        #    c. U, S, Vh = jnp.linalg.svd(Env, full_matrices=False)
                        # Env to matrix: group tensor indices into matrix indexes, separating upper from lower indices. TODO: check that grouping is correct. 
                        out_dims = Env.shape[Env.ndim//2:]
                        in_dims = Env.shape[:Env.ndim//2]
                        matrix_env = Env.reshape(np.prod(out_dims), np.prod(in_dims))
                        U, S, Vh = jnp.linalg.svd(matrix_env, full_matrices=False)
                        if svd_cutoff is not None:
                            U, S, Vh, k_trunc = compress_SVD(U, S, Vh, cutoff=svd_cutoff)
                        #    d. Apply cutoff to S if needed
                        
                        #    e. g_i.tensor = U @ Vh # Update gate tensor IN PLACE in the circuit object
                        new_gate_matrix = U @ Vh 
                        new_gate = gate.copy()
                        new_gate.matrix = new_gate_matrix
                        # Reshape back to original tensor shape
                        # new_gate_tensor = new_gate_matrix.reshape(Env.shape) # Reshape to Env shape

                        # Store the updated tensor temporarily
                        # updated_gate_tensors[gate_obj] = new_gate_tensor

                        #    f. Update E_left_current by contracting with the *updated* g_i, E_top_l sites, E_bottom_current sites
                        E_left_current, _ = _update_left_env(E_left_current, E_top_l[i], E_bottom_current[i], new_gate, E_top_l[i+1], E_bottom_current[i+1])
                        current_layer.gates[gate_idx_in_layer] = new_gate
                        gate_idx_in_layer += 1               
                        i +=2  
                        loss_history.append(trace)
    
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

                        current_layer.gates[gate_idx_in_layer] = new_gate
                        gate_idx_in_layer += 1
                        i += 1
                        loss_history.append(trace)

                    else: 
                        raise TypeError(
                            f"Unsupported gate object at layer {l}, site {i}: "
                            f"{gate!r} (expected single‑qubit, two‑qubit, or None)."
                        )
                
                # --- Optional: Right-to-Left Pass within layer ---
                # Similar logic, compute left boundaries, sweep right-to-left

            # --- Update the actual gates in the circuit layer ---
            # print(f"    Updating gates in circuit layer {l}...")
            # for gate_obj in circuit.layers[l].gates:
            #     if gate_obj in updated_gate_tensors:
            #         # Update the tensor. Assumes Gate.tensor is mutable or
            #         # the Circuit structure handles replacement correctly.
            #         # Using JAX arrays usually means creating a new object.
            #         # This depends heavily on your Circuit/Gate implementation.
            #         # Example: circuit.update_gate_tensor(l, gate_obj, updated_gate_tensors[gate_obj])
            #         gate_obj.tensor = updated_gate_tensors[gate_obj]
            #         # Re-run post_init if tensor needs update
            #         if hasattr(gate_obj, '__post_init__'): gate_obj.__post_init__()
            #     else:
            #             print(f"Warning: Gate {gate_obj.name} on {gate_obj.qubits} was not updated (maybe skipped).")

            # --- Update Bottom Environment for the *next* layer ---
                        
            # Only if not the last layer
            # TODO: update current layer in circuit Object
            # TODO: after an initial direction, alternate the contraction. 
            if l < circuit.num_layers - 1:
                E_bottom_current = contract_mpo_with_layer_left_to_right( # Choose L->R or R->L
                    E_bottom_current,
                    current_layer, # Use the *updated* layer
                    layer_is_below=False, # Contracting layer above E_bottom
                    max_bondim=max_bondim_env,

                    # Pass sweep_direction if needed
                )
                # Optionally alternate sweep_direction

        # --- Step 3: Optional Top-Down Sweep (Mirror of Step 2) ---
        # print("  Top-Down Sweep...")
        # E_top_current = mpo_ref_dag # Reset
        # Loop l from L-1 down to 0
        #    Retrieve/Compute E_bottom_l (might need separate caching/computation pass)
        #    Inner loop(s) for layer l update (R->L then L->R?)
        #    Update E_top_current based on layer l+1


        # --- End of Sweep ---
        # Calculate and print cost function
        # TODO: track trace.

    # return circuit with updated gates, return loss function. 
    return circuit, loss_history