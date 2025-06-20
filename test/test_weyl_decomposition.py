# test whether the weyl decomposed circuit matches the original circuit.
# test whether the absorbtion of single-qubit gates on the weyl decomposed circuit matches the original circuit.
import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotter_circuit, compress_consecutive_layers
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
import numpy as np
import jax.numpy as jnp

###############################################
# Test 1: # test whether the weyl decomposed circuit matches the original circuit.
###############################################

original_circ = trotter_circuit(
    n_sites=4,
    J=1.0,
    Delta=1.0,
    delta_t=0.1,
    n_steps=2,
    order=4,
    combine_gate_triplet=True,
    dtype=jnp.complex128
)
compress_consecutive_layers(original_circ)
original_circ.print_gates()


# original_circ is your 21-layer brickwall Circuit
expanded = weyl_decompose_circuit(original_circ)
expanded.print_gates(max_per_layer=4)   # should show 63 layers

orig_U   = jnp.array(original_circ.to_matrix())
expand_U = jnp.array(expanded.to_matrix())

# Two circuits are identical up to a global phase if
#    orig_U ≈ e^{iφ} expand_U
phi = jnp.angle(orig_U[0,0] / expand_U[0,0])
assert jnp.allclose(orig_U, jnp.exp(1j*phi)*expand_U, atol=1e-10)
print("✓ decomposition circuit reproduces the original")

###############################################
# Test 2: # test whether the absorbtion of single-qubit gates on the weyl decomposed circuit matches the original circuit.
###############################################

compressed = absorb_single_qubit_layers(expanded)   # `expanded` is the 3 N circuit
print(f"{expanded.num_layers=},   {compressed.num_layers=}")   # 63   43  (≡ 2·21+1)

# unitary equivalence (up to a global phase):
U_full   = jnp.array(expanded.to_matrix())
U_comp   = jnp.array(compressed.to_matrix())
phi      = jnp.angle(U_full[0,0] / U_comp[0,0])
assert jnp.allclose(U_full, jnp.exp(1j*phi)*U_comp, atol=1e-10)
print("✓ compressed circuit matches the full one.")
