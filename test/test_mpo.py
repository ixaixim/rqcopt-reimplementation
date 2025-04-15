from rqcopt_mpo.mpo.mpo_builder import get_id_mpo

# --- Example Usage ---
num_qubits = 4
id_mpo_obj = get_id_mpo(num_qubits)

print(f"Created Identity MPO for {id_mpo_obj.n_sites} sites.")
print(f"Physical dimensions (out, in): ({id_mpo_obj.physical_dim_out}, {id_mpo_obj.physical_dim_in})")
print(f"Number of tensors: {len(id_mpo_obj.tensors)}")
print(f"Shape of first tensor: {id_mpo_obj.tensors[0].shape}") # Should be (1, 2, 2, 1)
print(f"Shape of last tensor: {id_mpo_obj.tensors[-1].shape}") # Should be (1, 2, 2, 1)
