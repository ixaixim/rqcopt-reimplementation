import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.tensor_network.core_ops import canonicalize_local_tensor, merge_two_mpos_and_gate, split_tensor_into_half_canonical_mpo_pair, contract_mpo_with_layer_right_to_left, contract_mpo_with_layer_left_to_right
from rqcopt_mpo.core_objects import Gate, GateLayer
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.gradient import compute_layer_boundary_environments
# --- Example Usage ---
# Create a dummy 4D tensor (e.g., random)
# dummy_tensor = jnp.array(np.random.rand(3, 2, 2, 4), dtype=jnp.complex128)

# # Left canonicalize
# Q_left, R_right = canonicalize_local_tensor(dummy_tensor, mode='left')
# print("Left Canonicalization:")
# print("Q_left shape:", Q_left.shape) # Should be (3, 2, 2, new_r)
# print("R_right shape:", R_right.shape) # Should be (new_r, 4)
# # Check orthogonality (Q_left^dagger @ Q_left should be identity)
# matrix_q_left = Q_left.reshape(-1, Q_left.shape[-1])
# identity_check_left = matrix_q_left.conj().T @ matrix_q_left
# print("Identity check (left):", np.allclose(identity_check_left, np.eye(identity_check_left.shape[0])))
# # Check reconstruction
# reconstructed_left = jnp.einsum('...k,kl->...l', Q_left, R_right)
# print("Reconstruction check (left):", np.allclose(reconstructed_left, dummy_tensor))


# # Right canonicalize
# Q_right, R_left = canonicalize_local_tensor(dummy_tensor, mode='right')
# print("\nRight Canonicalization:")
# print("Q_right shape:", Q_right.shape) # Should be (new_l, 2, 2, 4)
# print("R_left shape:", R_left.shape) # Should be (3, new_l)
# # Check orthogonality (Q_right @ Q_right^dagger should be identity)
# matrix_q_right = Q_right.reshape(Q_right.shape[0], -1)
# identity_check_right = matrix_q_right @ matrix_q_right.conj().T
# print("Identity check (right):", np.allclose(identity_check_right, np.eye(identity_check_right.shape[0])))
# # Check reconstruction
# reconstructed_right = jnp.einsum('ik...,k...->i...', R_left, Q_right)
# print("Reconstruction check (right):", np.allclose(reconstructed_right, dummy_tensor))

# # --- Example Usage ---
# # Define dimensions
# l1, r1, r2 = 3, 4, 5
# p = 2 # physical dimension

# # Create dummy tensors
# dummy_mpo1 = jnp.array(np.random.rand(l1, p, p, r1), dtype=jnp.complex64)
# dummy_mpo2 = jnp.array(np.random.rand(r1, p, p, r2), dtype=jnp.complex64)
# dummy_gate = jnp.array(np.random.rand(p, p, p, p), dtype=jnp.complex64) # shape (p_out1, p_out2, p_in1, p_in2)

# # Test gate_on_left = True
# try:
#     merged_true = merge_two_mpos_and_gate(dummy_mpo1, dummy_mpo2, dummy_gate, gate_on_left=True)
#     # Expected shape: (l1, p, p, p, p, r2) -> (3, 2, 2, 2, 2, 5)
#     print(f"Merged (gate_on_left=True) shape: {merged_true.shape}")
# except ValueError as e:
#     print(f"Error (gate_on_left=True): {e}")


# # Test gate_on_left = False
# try:
#     merged_false = merge_two_mpos_and_gate(dummy_mpo1, dummy_mpo2, dummy_gate, gate_on_left=False)
#     # Expected shape: (l1, p, p, p, p, r2) -> (3, 2, 2, 2, 2, 5)
#     print(f"Merged (gate_on_left=False) shape: {merged_false.shape}")
# except ValueError as e:
#     print(f"Error (gate_on_left=False): {e}")


# --- Example Usage (assuming compress_SVD is defined) ---
# l_dim, p1o_dim, p2o_dim, p1i_dim, p2i_dim, r_dim = 3, 2, 2, 2, 2, 4
# dummy_merged_tensor = jnp.array(np.random.rand(l_dim, p1o_dim, p2o_dim, p1i_dim, p2i_dim, r_dim), dtype=jnp.complex128)

# max_chi = 10

# # Split with left canonicalization
# try:
#     mpo1_left, mpo2_left = split_tensor_into_half_canonical_mpo_pair(
#         dummy_merged_tensor, canonical_mode='left', max_bondim=max_chi
#     )
#     print("\nSplit (mode='left'):")
#     # Expected mpo1: (l, p1o, p1i, k) -> (3, 2, 2, k<=10)
#     # Expected mpo2: (k, p2o, p2i, r) -> (k<=10, 2, 2, 4)
#     print("mpo1 shape:", mpo1_left.shape)
#     print("mpo2 shape:", mpo2_left.shape)
#     # Check canonicalization of mpo1
#     matrix_mpo1_left = mpo1_left.reshape(-1, mpo1_left.shape[-1])
#     identity_check_mpo1 = matrix_mpo1_left.conj().T @ matrix_mpo1_left
#     print("mpo1 Identity check:", np.allclose(identity_check_mpo1, np.eye(identity_check_mpo1.shape[0]), atol=1e-7))

# except Exception as e:
#     print(f"Error during split (mode='left'): {e}")


# # Split with right canonicalization
# try:
#     mpo1_right, mpo2_right = split_tensor_into_half_canonical_mpo_pair(
#         dummy_merged_tensor, canonical_mode='right', max_bondim=max_chi
#     )
#     print("\nSplit (mode='right'):")
#     # Expected mpo1: (l, p1o, p1i, k) -> (3, 2, 2, k<=10)
#     # Expected mpo2: (k, p2o, p2i, r) -> (k<=10, 2, 2, 4)
#     print("mpo1 shape:", mpo1_right.shape)
#     print("mpo2 shape:", mpo2_right.shape)
#     # Check canonicalization of mpo2
#     matrix_mpo2_right = mpo2_right.reshape(mpo2_right.shape[0], -1)
#     identity_check_mpo2 = matrix_mpo2_right @ matrix_mpo2_right.conj().T
#     print("mpo2 Identity check:", np.allclose(identity_check_mpo2, np.eye(identity_check_mpo2.shape[0]), atol=1e-7))

# except Exception as e:
#      print(f"Error during split (mode='right'): {e}")
# # Check reconstruction (approximately, due to truncation)
# # Need to contract mpo1 and mpo2 back
# reconstruction = jnp.einsum('iabk,kcde->iabcde', mpo1_left, mpo2_left) # Or use mpo1_right, mpo2_right
# print("\nReconstruction check (approx):", np.allclose(reconstruction, merged_tensor, atol=1e-6)) # Use tolerance

# --- Example Usage ---
# Create dummy MPO and Layer
# n = 6
# phys_dim = 2
# bond_dim = 5
# mpo_middle_tensors = [jnp.array(np.random.rand(bond_dim, phys_dim, phys_dim, bond_dim)) for _ in range(4)]
# mpo_first_tensor = [jnp.array(np.random.rand(1, phys_dim, phys_dim, bond_dim))]
# mpo_last_tensor = [jnp.array(np.random.rand(bond_dim, phys_dim, phys_dim, 1))]

# mpo_t = mpo_first_tensor + mpo_middle_tensors + mpo_last_tensor

# for k in range(n-1): # Ensure bonds match
#     new_shape = mpo_t[k+1].shape
#     mpo_t[k+1] = jnp.array(np.random.rand(mpo_t[k].shape[-1], new_shape[1], new_shape[2], new_shape[3]))
# mpo_t[-1] = mpo_t[-1][..., :1] # Fix last bond dim

# dummy_mpo = MPO(tensors=mpo_t)

# # Layer with 1Q and 2Q gates
# gates = [
#     # Gate(jnp.array(np.random.rand(2,2)), qubits=(1,), layer_index=0), # 1Q on site 1
#     Gate(jnp.array(np.random.rand(4,4)), qubits=(2,3), layer_index=0), # 2Q on 2,3 # at the moment, this is being problematic. TO CHECK. 
#     Gate(jnp.array(np.random.rand(2,2)), qubits=(4,), layer_index=0), # 1Q on site 4
#     Gate(jnp.array(np.random.rand(4,4)), qubits=(0,1), layer_index=0), # 2Q on 0,1
# ]
# dummy_layer = GateLayer(gates=gates, layer_index=0, is_odd=None)

# # Contract (example: layer below MPO)
# try:
#     # contracted_mpo_rl = contract_mpo_with_layer_right_to_left(
#     #     dummy_mpo, dummy_layer, layer_is_below=True, max_bondim=10
#     # )
#     # print("\nRight-to-Left Contraction Successful.")
#     # print(f"Resulting MPO sites: {contracted_mpo_rl.n_sites}")
#     # print(f"Resulting MPO bond dimensions: {[t.shape[0] for t in contracted_mpo_rl.tensors[1:]]+[contracted_mpo_rl.tensors[-1].shape[-1]]}")
#     # print(f"Is right canonical (approx): {contracted_mpo_rl.is_right_canonical}")
#     contracted_mpo_rl = contract_mpo_with_layer_left_to_right(
#         dummy_mpo, dummy_layer, layer_is_below=True, max_bondim=10
#     )
#     print("\nLeft-to-Right Contraction Successful.")
#     print(f"Resulting MPO sites: {contracted_mpo_rl.n_sites}")
#     print(f"Resulting MPO bond dimensions: {[t.shape[0] for t in contracted_mpo_rl.tensors[1:]]+[contracted_mpo_rl.tensors[-1].shape[-1]]}")
#     print(f"Is left canonical (approx): {contracted_mpo_rl.is_left_canonical}")


#     # Check canonicalization (optional, might fail slightly due to precision/truncation)
#     # mps_like = [t.reshape(t.shape[0],-1) for t in contracted_mpo_rl.tensors]
#     # for k in range(len(mps_like)-1,0,-1):
#     #      mm_dag = mps_like[k] @ mps_like[k].conj().T
#     #      print(f"Site {k} identity check:", np.allclose(mm_dag, np.eye(mm_dag.shape[0]), atol=1e-6))

# except Exception as e:
#      print(f"\nError during Right-to-Left Contraction: {e}")
#      import traceback
#      traceback.print_exc()


# --- Example Setup ---
n_sites = 5
phys_dim = 2
bond_dim_top = 4 # Example bond dims
bond_dim_bot = 3
dtype = jnp.complex64

def create_dummy_mpo(n, bond_dim, dtype):
    tensors = []
    # First tensor
    l, r = 1, bond_dim
    tensors.append(jnp.array(np.random.rand(l, phys_dim, phys_dim, r) + 1j*np.random.rand(l, phys_dim, phys_dim, r), dtype=dtype))
    # Middle tensors
    for _ in range(n - 2):
        l, r = bond_dim, bond_dim
        tensors.append(jnp.array(np.random.rand(l, phys_dim, phys_dim, r) + 1j*np.random.rand(l, phys_dim, phys_dim, r), dtype=dtype))
    # Last tensor
    l, r = bond_dim, 1
    tensors.append(jnp.array(np.random.rand(l, phys_dim, phys_dim, r) + 1j*np.random.rand(l, phys_dim, phys_dim, r), dtype=dtype))
    # Ensure bonds match (simple fix for example)
    for k in range(n-1):
         shape_next = tensors[k+1].shape
         tensors[k+1] = tensors[k+1][:,:,:,:tensors[k].shape[-1]].transpose(3,1,2,0)[:shape_next[0],:,:,:] # Crude matching
         tensors[k+1] = jnp.array(np.random.rand(tensors[k].shape[-1], shape_next[1], shape_next[2], shape_next[3]) + 1j*np.random.rand(tensors[k].shape[-1], shape_next[1], shape_next[2], shape_next[3]), dtype=dtype)
    tensors[-1] = tensors[-1][...,:1]

    return MPO(tensors=tensors)

# Create MPOs
E_top = create_dummy_mpo(n_sites, bond_dim_top, dtype)
E_bottom = create_dummy_mpo(n_sites, bond_dim_bot, dtype)
print(f"Created E_top with {E_top.n_sites} sites.")
print(f"Created E_bottom with {E_bottom.n_sites} sites.")

# Create Layer with Gates
gates = [
    # Correctly use 4D tensor for 2Q gate for Gate.__post_init__
    Gate(jnp.array(np.random.rand(4,4) + 1j*np.random.rand(4,4), dtype=dtype), qubits=(0,1), layer_index=1, name="G01"),
    Gate(jnp.array(np.random.rand(4,4) + 1j*np.random.rand(4,4), dtype=dtype), qubits=(2,3), layer_index=1, name="G23"),
    # Gate(jnp.array(np.random.rand(2,2,2,2) + 1j*np.random.rand(2,2,2,2), dtype=dtype), qubits=(2,3), layer_index=1, name="G23"), # Will overwrite G3 in right map
    Gate(jnp.array(np.random.rand(2,2) + 1j*np.random.rand(2,2), dtype=dtype), qubits=(4,), layer_index=1, name="G4"),
]
dummy_layer = GateLayer(gates=gates, layer_index=1, is_odd=False)
print(f"Created dummy layer with {len(dummy_layer.gates)} gates.")

# --- Run the Function ---

# Test Left Sweep
print("\n--- Testing Left Sweep ---")
try:
    # Assuming E_top, E_bottom, dummy_layer are defined as in the previous example
    left_envs = compute_layer_boundary_environments(
        E_top, E_bottom, dummy_layer, side='left'
    )
    print("Left sweep executed successfully.")
    print(f"Length of returned list: {len(left_envs)} (should be n_sites={n_sites})")

    # Count and display non-None environments
    num_found_envs = 0
    for i in range(n_sites): # Iterate through potential starting sites
        env = left_envs[i]
        if env is not None:
            num_found_envs += 1
            # Describe what the environment represents
            print(f"  Shape of env left of gate starting at site {i} (left_envs[{i}]): {env.shape}")
        # else:
            # Optionally print sites with no env stored
            # print(f"  No environment stored for site {i}.")

    print(f"Total non-None left environments found: {num_found_envs}")

    # Verify against expected number of gates starting within bounds
    expected_envs = 0
    gate_map_left_test = {i: None for i in range(n_sites)} # Recreate for test scope
    for gate in dummy_layer.gates:
         leftmost_site = min(gate.qubits)
         if leftmost_site < n_sites:
             gate_map_left_test[leftmost_site] = gate # Doesn't matter which gate, just that one starts here
    expected_envs = sum(1 for g in gate_map_left_test.values() if g is not None)

    print(f"Expected number of left environments based on gate map: {expected_envs}")
    if num_found_envs != expected_envs:
        print("WARNING: Number of found environments does not match the expected number!")

except Exception as e:
    print(f"ERROR during left sweep test: {e}")
    import traceback
    traceback.print_exc()

# Test Right Sweep
print("\n--- Testing Right Sweep ---")
try:
    # Assuming E_top, E_bottom, dummy_layer are defined as in the previous example
    right_envs = compute_layer_boundary_environments(
        E_top, E_bottom, dummy_layer, side='right'
    )
    print("Right sweep executed successfully.")
    print(f"Length of returned list: {len(right_envs)} (should be n_sites={n_sites})")

    # Count and display non-None environments
    num_found_envs = 0
    for i in range(n_sites):
        env = right_envs[i]
        if env is not None:
            num_found_envs += 1
            # Describe what the environment represents
            print(f"  Shape of env right of gate ending at site {i} (right_envs[{i}]): {env.shape}")
        # else:
            # Optionally print sites with no env stored
            # print(f"  No environment stored for site {i}.")

    print(f"Total non-None right environments found: {num_found_envs}")

    # Verify against expected number of gates ending within bounds
    expected_envs = 0
    gate_map_right_test = {i: None for i in range(n_sites)} # Recreate for test scope
    for gate in dummy_layer.gates:
         rightmost_site = max(gate.qubits)
         if rightmost_site < n_sites:
             gate_map_right_test[rightmost_site] = gate # Doesn't matter which gate, just that one ends here
    expected_envs = sum(1 for g in gate_map_right_test.values() if g is not None)

    print(f"Expected number of right environments based on gate map: {expected_envs}")
    if num_found_envs != expected_envs:
        print("WARNING: Number of found environments does not match the expected number!")

except Exception as e:
    print(f"ERROR during right sweep test: {e}")
    import traceback
    traceback.print_exc()

