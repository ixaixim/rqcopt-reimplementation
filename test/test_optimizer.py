from rqcopt_mpo.mpo.mpo_builder import create_dummy_mpo

# create dummy reference mpo
# original numbers
n_sites     = 5
phys_dim    = 2
bond_left_t,  bond_mid_t,  bond_right_t  = 2, 5, 4

# right bond dimension per site (len == n_sites)
bond_dims    = [2, 5, 4, 1, 1]

reference_mpo    = create_dummy_mpo(bond_dims, phys_dim, random=True, seed=42)

