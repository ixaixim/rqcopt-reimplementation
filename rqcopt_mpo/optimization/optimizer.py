from typing import Optional, List, Dict
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.core_objects import Circuit, GateLayer, Gate 
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.mpo.mpo_builder import get_id_mpo
from rqcopt_mpo.tensor_network import contract_mpo_with_layer_left_to_right, contract_mpo_with_layer_right_to_left
from rqcopt_mpo.optimization.gradient import compute_layer_boundary_environments, _update_left_env, _update_right_env, gate_map, compute_gate_environment_tensor, compute_trace
# TODO: create a compute_full_bottom_environment function. to place in gradient.py, instead of doing it optimizer.py


def optimize_circuit_local_svd(
    circuit_initial: Circuit,
    mpo_ref: MPO,
    num_sweeps: int,
    max_bondim_env: int,           # Max bond dim for environment MPOs
    svd_cutoff: float = 1e-10,     # Cutoff for gate update SVD
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

    # 2. Get reference MPO dagger (needs implementation in MPO class)
    mpo_ref_dag = mpo_ref.copy()
    # TODO?
    # mpo_ref_dag = mpo_ref.dagger()
    # 3. Get identity MPO
    mpo_identity = get_id_mpo(circuit.n_sites, dtype=mpo_ref.tensors[0].dtype)

    # 4. Initialize environment caches 
    #    Using dictionaries keyed by layer index might be flexible.
    all_E_top: Dict[int, MPO] = {}
    all_E_bottom: Dict[int, MPO] = {}

    # Loss tracking 
    loss_history = []

    # --- Main Sweep Loop ---
    for sweep in range(num_sweeps):
        print(f"\n--- Starting Sweep {sweep + 1}/{num_sweeps} ---")

        # --- Optional: MPO Environment Reset ---
        # if mpo_reset_freq is not None and sweep % mpo_reset_freq == 0:
            # print("  Resetting environment MPOs...")
            # Reset logic based on mpo_ref_dag and mpo_identity
            # Force recomputation in the first steps of the sweeps below


        # --- Step 1: Compute/Cache All Top Environments (Top-to-Bottom) ---
        print("  Computing Top Environments...")
        E_top_current = mpo_ref_dag # Or retrieve from reset cache
        all_E_top = {circuit.num_layers-1: E_top_current} # Assuming L layers, index L is above layer L-1
        sweep_direction = 'left_to_right' # Or 'right_to_left', choose convention 
        for l in range(circuit.num_layers - 2, 0, -1): # L-1 down to 1
            layer_above = circuit.layers[l+1] # The layer just processed

            # Pick the correct contraction routine for this step
            if sweep_direction == "left_to_right":
                E_top_current = contract_mpo_with_layer_left_to_right(
                    E_top_current,
                    layer_above,
                    layer_is_below=True,          # contracting a layer *below* E_top
                    max_bondim=max_bondim_env,
                )
                sweep_direction = "right_to_left"  # flip for the next layer
            else:
                E_top_current = contract_mpo_with_layer_right_to_left(
                    E_top_current,
                    layer_above,
                    layer_is_below=True,
                    max_bondim=max_bondim_env,
                )
                sweep_direction = "left_to_right"  # flip back for the next layer
            # Cache environment that now sits *above* layer l
            all_E_top[l] = E_top_current         
            print(f"    Stored E_top[{l}]")
       

        # --- Step 2: Bottom-Up Sweep (Update Gates) ---
        print("  Bottom-Up Sweep...")
        E_bottom_current = mpo_identity 
        sweep_direction = 'left_to_right' # Start opposite? Choose convention # direction of canonicalization as we contract layer with bottom MPO. 
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
                gate_map_left, gate_map_right = gate_map(current_layer, n_sites)
                # cache left env only if a gate starts at the site
                if gate_map_left.get(0) is not None:
                    E_left_boundaries[0] = E_left_current # Environment left of site 0
                stop_index = max(s for s, g in gate_map_left.items() if g is not None)

                while i < stop_index:
                    # if gate starting at i is present, we check if it is single or two-qubit, or None
                    gate = gate_map_left.get(i)
                    if gate.is_two_qubit():

                    #    a. Retrieve E_right_i from E_right_boundaries
                        right_site_index = max(gate.qubits)
                        E_right_current = E_right_boundaries[right_site_index]
                        if E_right_current is None:
                            print(f"Error: Missing right environment for gate ending at site {right_site_index}. Skipping update.")
                        # Update left environment assuming no gate interaction for this step

                    #    b. Env = compute_gate_environment_tensor(g_i.qubits, E_top_l, E_bottom_current, E_left_current, E_right_i) #TODO: in gradient.py
                        
                        Env = compute_gate_environment_tensor(gate.qubits, E_top_l, E_bottom_current, E_left_current, E_right_current)
                        trace = compute_trace(Env, gate.tensor_4d)

                    #    c. U, S, Vh = jnp.linalg.svd(Env, full_matrices=False)
                        # Env to matrix: group tensor indices into matrix indexes, separating upper from lower indices. TODO: check that grouping is correct. 
                        out_dims = Env.shape[:Env.ndim//2]
                        in_dims = Env.shape[Env.ndim//2:]
                        matrix_env = Env.reshape(np.prod(out_dims), np.prod(in_dims))
                        U, S, Vh = jnp.linalg.svd(matrix_env, full_matrices=False)

                    #    d. Apply cutoff to S if needed
                        
                    #    e. g_i.tensor = U @ Vh # Update gate tensor IN PLACE in the circuit object
                        # TODO: check if we can reassign arrays with jax. 
                        new_gate_matrix = U @ Vh 
                        # Reshape back to original tensor shape
                        new_gate_tensor = new_gate_matrix.reshape(Env.shape) # Reshape to Env shape

                        # Store the updated tensor temporarily
                        # updated_gate_tensors[gate_obj] = new_gate_tensor

                        # TODO: assign gate to circuit layer. 
                    #    f. Update E_left_current by contracting with the *updated* g_i, E_top_l sites, E_bottom_current sites
                        E_left_current = _update_left_env(E_left_current, E_top_l[i], E_bottom_current[i], new_gate_tensor, E_top_l[+1], E_bottom_current[+1])

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
            #         # Re-run post_init if tensor_4d needs update
            #         if hasattr(gate_obj, '__post_init__'): gate_obj.__post_init__()
            #     else:
            #             print(f"Warning: Gate {gate_obj.name} on {gate_obj.qubits} was not updated (maybe skipped).")

            # --- Update Bottom Environment for the *next* layer ---
                        
            # Only if not the last layer
            # TODO: update current layer with updated gates.
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

    # return circuit
    pass # Replace with actual return