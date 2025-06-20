# test whether the MPO is fully normalized. 
import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_builder import _random_unitary, generate_random_circuit
from rqcopt_mpo.circuit.circuit_dataclasses import Gate
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax 


n_sites = 6
# generate random target circuit.
U_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=25,
    p_single=0.3,
    p_two=0.4,
    seed=43,
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

U_circuit.sort_layers()
U_circuit.print_gates()
# transform target into MPO
print("Converting target circuit to MPO...")
U = circuit_to_mpo(U_circuit, svd_cutoff=0.0) 
U.left_canonicalize()
U.normalize()

U_dag = U.dagger()
L_env = jnp.eye(1).reshape(1,1)
for i in range(n_sites):
    L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, U[i], U_dag[i])

mpo_product = L_env[0,0]
print(f"Norm after normalization: |Tr(U^dag U)|^2: {mpo_product}")
# TODO: assert it matches 1

