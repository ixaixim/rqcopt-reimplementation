# verify how the HIlbert Schmidt scalar product is computed in practice when A and B are MPOs instead of matrices. 
import rqcopt_mpo.jax_config

from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
import jax.numpy as jnp
import numpy as np
import jax


n_sites = 4
seed_A = 42
seed_B = 43
# create unitary matrix A from circuit
    # generate random target circuit.
A_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=3,
    p_single=0.3,
    p_two=0.3,
    seed=seed_A,
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

A_circuit.sort_layers()
A_circuit.print_gates()
    # transform target into MPO
print("Converting A circuit to MPO...")
A_MPO = circuit_to_mpo(A_circuit)
A_MPO.left_canonicalize()
A_matrix = A_circuit.to_matrix()

# create unitary matrix B from circuit
B_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=3,
    p_single=0.3,
    p_two=0.3,
    seed=seed_B,
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

B_circuit.sort_layers()
B_circuit.print_gates()
    # transform target into MPO
print("Converting B circuit to MPO...")
B_MPO = circuit_to_mpo(B_circuit)
B_MPO.left_canonicalize()
B_matrix = B_circuit.to_matrix()

# compute Tr(A^dag B) explicitly. 
explicit = np.trace(np.conjugate(A_matrix).T @ B_matrix)
print(f"explicit scalar product: {explicit}")

# compute Tr(A^dag B) with MPO.
A_MPO = A_MPO.dagger()
L_env = jnp.eye(1).reshape(1,1)
for i in range(n_sites):
    L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, A_MPO[i], B_MPO[i])

mpo_product = L_env[0,0]
print(f"MPO product: {mpo_product}")