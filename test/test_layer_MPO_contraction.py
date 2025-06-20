import rqcopt_mpo.jax_config

from rqcopt_mpo.mpo.mpo_builder import create_dummy_mpo
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer
from rqcopt_mpo.utils.utils import gate_map, layer_to_matrix
from rqcopt_mpo.circuit.circuit_builder import _random_unitary
from rqcopt_mpo.tensor_network.core_ops import contract_mpo_with_layer
import jax
import jax.numpy as jnp
import numpy as np

######################################
# Test the canonical form of the MPO after contraction: every site must be left/right isometry.
######################################
seed_layer = 52
seed_mpo = 42
layer_is_below=True
n_sites = 5

# create random circuit layer with None gates in between.
key_pool  = (jax.random.split(jax.random.PRNGKey(seed_layer), 3))
gate0 = None
gate12 = Gate(matrix=_random_unitary(dim=4, key=key_pool[0], dtype=jnp.complex128), qubits=(1,2), layer_index=0)
gate_3 = None
gate4 = Gate(matrix=_random_unitary(dim=2, key=key_pool[1]), qubits=(4,), layer_index=0)

layer = GateLayer(layer_index=0, gates=[gate12, gate4], is_odd=True)


#####################################
# create random MPO 
bond_dims_right = [2,4,5,4,1] # 5 sites
assert len(bond_dims_right) == n_sites, f"Expected length {n_sites}, but got {len(bond_dims_right)}"
mpo = create_dummy_mpo(bond_dims_right=bond_dims_right, random=True, seed=seed_mpo, dtype=jnp.complex128)
mpo.left_canonicalize() # left to right QR sweep.

# contract layer with MPO right to left
direction = 'right_to_left' # right to left RQ sweep just after contraction.
final_mpo = contract_mpo_with_layer(mpo_init=mpo, layer=layer, layer_is_below=layer_is_below, direction='right_to_left')

# check every site is right isometry except last one
for site in final_mpo[1:]:

    shape = site.shape
    site_matrix = site.reshape(shape[0], -1)
    right_product =  site_matrix @ site_matrix.conj().T
    I = np.eye(right_product.shape[0], dtype=site_matrix.dtype)
    assert np.allclose(right_product, I, atol=1e-8), f"Matrix differs from identity by max {np.max(np.abs(right_product - I))}"
print("All sites are right isometries up to the first one.")

####################################
# create random MPO 
bond_dims_right = [2,4,5,4,1] # 5 sites
assert len(bond_dims_right) == n_sites, f"Expected length {n_sites}, but got {len(bond_dims_right)}"
mpo = create_dummy_mpo(bond_dims_right=bond_dims_right, random=True, seed=seed_mpo, dtype=jnp.complex128)
mpo.right_canonicalize() 

# contract layer with MPO left to right
final_mpo = contract_mpo_with_layer(mpo_init=mpo, layer=layer, layer_is_below=layer_is_below, direction='left_to_right')
# check every site is left isometry except first one
for site in final_mpo[:-1]:

    shape = site.shape
    site_matrix = site.reshape(-1, shape[-1])
    left_product = site_matrix.conj().T @ site_matrix
    I = np.eye(left_product.shape[0], dtype=site_matrix.dtype)
    assert np.allclose(left_product, I, atol=1e-8), f"Matrix differs from identity by max {np.max(np.abs(left_product - I))}"
print("All sites are left isometries up to the last one.")


######################################
# Test the reconstruction of the matrix. 
######################################

# MPO to matrix
mpo_matrix = mpo.to_matrix()
final_mpo_matrix = final_mpo.to_matrix()
print()
print(f'Successfully created matrix from MPO of shape {final_mpo_matrix.shape}')

# layer to matrix
layer_matrix = layer_to_matrix(layer=layer, n_sites=n_sites)

print(f'Successfully created matrix from layer of shape {layer_matrix.shape}')
# Matrix_MPO @ Matrix_layer
if layer_is_below: # layer is temporally placed before
    prod = mpo_matrix @ layer_matrix
else: 
    prod = layer_matrix @ mpo_matrix

# check that product matches:
np.testing.assert_allclose(final_mpo_matrix, prod, atol=1e-14), f"Max diff = {jnp.max(jnp.abs(final_mpo_matrix - prod))}"
