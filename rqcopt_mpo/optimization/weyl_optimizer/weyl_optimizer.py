import rqcopt_mpo.jax_config

# NOTE: this is a modified copy of optimizer.py. Verify that the two versions are the same, except for the 2 Qubit update. 
from typing import Optional, List, Dict
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, GateLayer
from qiskit.synthesis import TwoQubitWeylDecomposition
from rqcopt_mpo.hamiltonian.operators import two_qubit_paulis
from jax.scipy.linalg import expm

from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.mpo.mpo_builder import get_id_mpo
from rqcopt_mpo.tensor_network.core_ops import contract_mpo_with_layer 
from rqcopt_mpo.optimization.gradient import compute_layer_boundary_environments, _update_left_env, _update_right_env, gate_map, compute_gate_environment_tensor, compute_trace, compute_upper_lower_environments
from rqcopt_mpo.optimization.utils import global_loss


# TODO: cleaner code would never let gate == None, if there is no gate at a site. Cleaner code would just use a single qubit identity gate. Then we can drop the None handling. 
#       This and lengthy syntax could be avoided if we had an iterator on the layer gates.
def optimize_weyl_circuit_local_svd(
    circuit_initial: Circuit,
    mpo_ref: MPO,
    num_sweeps: int,
    max_bondim_env: int,           # Max bond dim for environment MPOs
    layer_update_passes: int = 1,  # Passes within a layer (L<->R)
    svd_cutoff: float = 1e-12,
    target_is_normalized : bool = False,
) -> Circuit:
    """
    Optimizes a quantum circuit using iterative local SVD updates based on
    environments computed against a reference MPO. Implements the sweep
    strategy described in Gibbs et al., arXiv:2409.16361v1.
    Adapted to Weyl circuit. The 2-qubit gates are SVD'd, then updated Weyl-like.

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
            mpo_ref=mpo_ref, circuit=circuit, direction='top', init_direction=top_layer_sweep_direction, 
            max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff,
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
                    n_sites, dtype,
                )
                loss_history.extend(pass_losses_lr)
                print(global_loss(pass_losses_lr, n_sites, target_is_normalized))
                # --- Right-to-Left Pass within layer ---
                # Similar logic, compute left boundaries, sweep right-to-left
                # NOTE: the logic of the R-> L pass is different from the L->R, in the sense that it is more simple.
                pass_losses_rl = _layer_pass_right_to_left(
                    current_layer, E_top_l, E_bottom_current, 
                    n_sites, dtype,
                )
                loss_history.extend(pass_losses_rl)
                print(global_loss(pass_losses_rl, n_sites, target_is_normalized))


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
                    n_sites, dtype, 
                )
                loss_history.extend(pass_losses_lr)
                print(global_loss(pass_losses_lr, n_sites, target_is_normalized))
                # --- Right-to-Left Pass within layer ---
                # Similar logic, compute left boundaries, sweep right-to-left
                # NOTE: the logic of the R-> L pass is different from the L->R, in the sense that it is more simple.
                pass_losses_rl = _layer_pass_right_to_left(
                    current_layer, E_top_current, E_bottom_l, 
                    n_sites, dtype,
                )
                loss_history.extend(pass_losses_rl)
                print(global_loss(pass_losses_rl, n_sites, target_is_normalized))

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

    return circuit, loss_history

def _layer_pass_left_to_right(
    current_layer: GateLayer,        # <-- mutated in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
) -> List[float]:
    """
    Left-to-right sweep through one GateLayer.
    Updates current_layer.gates in-place and returns the sweep loss history.
    """
    print("      L->R Pass...")
    pass_loss_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1.  Pre–compute all *right* boundary environments for the layer
    #     (the mirror of the “left” boundaries used in the R→L sweep).
    # ──────────────────────────────────────────────────────────────────────────
    E_right_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side="right"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 2.  Initialise the running *left* environment to the identity (bond-dim 1)
    # ──────────────────────────────────────────────────────────────────────────
    E_left_current = jnp.eye(1, dtype=dtype)

    # Build a dict {site_idx : gate | None} for O(1) lookup.
    gate_map_left, _ = gate_map(current_layer, n_sites)

    # No need to walk past the right-most gate in the layer.
    last_gate_at_idx = max(s for s, g in gate_map_left.items() if g is not None)

    # We also keep an id→index map so we can overwrite the *exact* object
    # in current_layer.gates after the SVD step.
    gate_to_idx_map = {id(g): i for i, g in enumerate(current_layer.gates)}

    # ──────────────────────────────────────────────────────────────────────────
    # 3.  Sweep from the far-left MPO tensor (site 0) to the last gate.
    # ──────────────────────────────────────────────────────────────────────────
    site_i = 0
    while site_i <= last_gate_at_idx:

        gate = gate_map_left.get(site_i)

        # ──────────────────────────────────────────────────────────────────
        # 3a.  No gate starting at *site_i*  →  propagate identity one site
        # ──────────────────────────────────────────────────────────────────
        if gate is None:
            E_left_current, step = _update_left_env(
                E_left_current,
                E_top_l[site_i], E_bottom_current[site_i],
            )
            site_i += step                       # step == 1 for identity hop
            continue

        # ──────────────────────────────────────────────────────────────────
        # 3b.  Gate present.  Gather its right environment and build Env.
        # ──────────────────────────────────────────────────────────────────
        rightmost_qb = max(gate.qubits)          # site index of the gate’s RHS
        E_right_current = E_right_boundaries[rightmost_qb]
        if E_right_current is None:
            print(
                f"Error: Missing left-hand env for gate ending at site "
                f"{rightmost_qb}.  Skipping update."
            )
            # Still need to move past the gate even if we skip, else infinite loop
            site_i += (2 if gate.is_two_qubit() else 1)
            continue

        Env = compute_gate_environment_tensor(
            gate.qubits,
            E_top_l, E_bottom_current,
            E_left_current,                     # already accumulated left env
            E_right_current                     # pre-computed right env
        )

        # ──────────────────────────────────────────────────────────────────
        # 3c.  SVD-based polar-project update  (same recipe as R→L pass)
        # ──────────────────────────────────────────────────────────────────
        env_ndim     = Env.ndim
        out_shape    = Env.shape[env_ndim // 2 :]
        in_shape     = Env.shape[: env_ndim // 2]
        matrix_env   = Env.reshape(np.prod(out_shape), np.prod(in_shape))

        U, S, Vh = jnp.linalg.svd(matrix_env, full_matrices=False)

        new_gate_matrix = _weyl_gate_update(U, Vh)
        new_gate_obj    = gate.copy()            # retain meta-data/qubits
        new_gate_obj.matrix = new_gate_matrix

        trace = compute_trace(Env, new_gate_obj.tensor)
        pass_loss_history.append(trace)

        # Overwrite the gate inside current_layer.gates
        idx_in_layer_list = gate_to_idx_map[id(gate)]
        current_layer.gates[idx_in_layer_list] = new_gate_obj

        # ──────────────────────────────────────────────────────────────────
        # 3d.  Push the *updated* gate into the running left environment
        # ──────────────────────────────────────────────────────────────────
        if gate.is_two_qubit():
            # Need both sites: site_i and rightmost_qb == site_i+1
            E_left_current, step = _update_left_env(
                E_left_current,
                E_top_l[site_i],   E_bottom_current[site_i],
                new_gate_obj,
                E_top_l[rightmost_qb], E_bottom_current[rightmost_qb]
            )
        else:  # single-qubit gate
            E_left_current, step = _update_left_env(
                E_left_current,
                E_top_l[site_i],   E_bottom_current[site_i],
                new_gate_obj
            )

        # ──────────────────────────────────────────────────────────────────
        # 3e.  Advance the site index by 1 (single-qubit) or 2 (two-qubit)
        # ──────────────────────────────────────────────────────────────────
        site_i += step   # step is guaranteed to be 1 or 2 from _update_left_env

    return pass_loss_history

# TODO: need to re use the L/R environments from the previous pass!
def _layer_pass_right_to_left(
    current_layer: GateLayer,  # Modified in-place
    E_top_l: MPO,
    E_bottom_current: MPO,
    n_sites: int,
    dtype: jnp.dtype,
) -> List[float]:

    print("      R->L Pass...")
    pass_loss_history = []

    E_left_boundaries = compute_layer_boundary_environments(
        E_top_l, E_bottom_current, current_layer, side='left'
    )
    gate_to_idx_map = {id(g): i for i, g in enumerate(current_layer.gates)}
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
            # Env tensor dimensions are assumed: (in_0, in_1, ..., out_0, out_1, ...)
            env_ndim = Env.ndim
            out_indices_shape = Env.shape[env_ndim//2:] # Shape of output physical legs (for the 'ket' part of the gate matrix)
            in_indices_shape = Env.shape[:env_ndim//2] # Shape of input physical legs (for the 'bra' part of the gate matrix)

            
            matrix_env = Env.reshape(np.prod(out_indices_shape), np.prod(in_indices_shape))
            U, S, Vh = jnp.linalg.svd(matrix_env, full_matrices=False)
            
            new_gate_matrix =_weyl_gate_update(U,Vh) # Shape (TotalOutDim, TotalInDim)
            new_gate_obj = gate.copy() # Copies structure (qubits, name, etc.)
            new_gate_obj.matrix = new_gate_matrix 

            trace = compute_trace(Env, new_gate_obj.tensor)
            pass_loss_history.append(trace)

            idx_in_layer_list = gate_to_idx_map[id(gate)]
            current_layer.gates[idx_in_layer_list] = new_gate_obj

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

def _weyl_gate_update(U: jnp.ndarray, Vh: jnp.ndarray) -> jnp.ndarray:
    update = U @ Vh
    dtype = U.dtype
    # if gate is single qubit, do regular update:
    if update.shape == (2,2):
        return update
    elif update.shape == (4,4):
        decomp = TwoQubitWeylDecomposition(update)
        a, b, c = decomp.a, decomp.b, decomp.c
        gl_phase = jnp.exp(1j * decomp.global_phase) # not essential for compression purposes, but used to test the match of the weyl decomposed circuit with the original circuit

        # (b) e^{i(...)} layer ---------------------------------------------------
        XX, YY, ZZ = two_qubit_paulis(dtype=dtype)
        nonlocal_op = gl_phase * expm(1j*(a*XX + b*YY + c*ZZ))
        return nonlocal_op

    else:
        raise NotImplementedError(f"Gate shape is {update.shape}. Not supported at the moment.")