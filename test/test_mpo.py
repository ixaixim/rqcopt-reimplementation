from rqcopt_mpo.mpo.mpo_builder import get_id_mpo, create_dummy_mpo

# --- Example Usage ---
num_qubits = 4
id_mpo_obj = get_id_mpo(num_qubits)

print(f"Created Identity MPO for {id_mpo_obj.n_sites} sites.")
print(f"Physical dimensions (out, in): ({id_mpo_obj.physical_dim_out}, {id_mpo_obj.physical_dim_in})")
print(f"Number of tensors: {len(id_mpo_obj.tensors)}")
print(f"Shape of first tensor: {id_mpo_obj.tensors[0].shape}") # Should be (1, 2, 2, 1)
print(f"Shape of last tensor: {id_mpo_obj.tensors[-1].shape}") # Should be (1, 2, 2, 1)

# original numbers
n_sites     = 5
phys_dim    = 2
bond_left_t,  bond_mid_t,  bond_right_t  = 2, 5, 4
bond_left_b,  bond_mid_b,  bond_right_b  = 3, 6, 5

# right bond dimension per site (len == n_sites)
bond_dims_top    = [bond_left_t,  bond_mid_t,  bond_right_t, 1, 1]
bond_dims_bottom = [bond_left_b,  bond_mid_b,  bond_right_b, 1, 1]

E_top_layer    = create_dummy_mpo(bond_dims_top,    phys_dim, random=True, seed=42)
E_bottom_layer = create_dummy_mpo(bond_dims_bottom, phys_dim, random=True, seed=42)