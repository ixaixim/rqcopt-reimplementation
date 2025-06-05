# create two shallow circuits U and W^ref, with same layout. Match U with W^ref
# test whether the optimization routine works when MPO and init circuit represent the same unitary.
# expected behavior: cost should drive down to zero. 
import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_builder import _random_unitary, generate_random_circuit
from rqcopt_mpo.circuit.circuit_dataclasses import Gate
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax 

n_sites = 4
target_seed = 44
# generate random target circuit.
target_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=8,
    p_single=0.3,
    p_two=0.3,
    seed=45, # for gate matrix
    rng=np.random.default_rng(seed=target_seed),
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

target_circuit.sort_layers()
target_circuit.print_gates()
# transform target into MPO
print("Converting target circuit to MPO...")
target_mpo = circuit_to_mpo(target_circuit)
target_mpo.left_canonicalize()


# generate initial circuit with same layout, but different gates.
init_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=8,
    p_single=0.3,
    p_two=0.3,
    seed=43,
    rng=np.random.default_rng(seed=target_seed),
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

print("Generated initial circuit with the same layout but different gates as the target circuit:")
init_circuit.sort_layers()
init_circuit.print_gates()


# run optimization routine. 
_, loss = optimize_circuit_local_svd(circuit_initial=init_circuit, mpo_ref=target_mpo, num_sweeps=1, layer_update_passes=1, max_bondim_env=128, svd_cutoff=1e-14)

loss_hst = 1 - 1/2**(2*n_sites) * np.abs(loss)**2
# if jnp.allclose(jnp.array(loss_hst), 0, atol=1e-12):
#     print("Result is close to zero within atol=1e-12.")
# else:
#     print("Result is NOT close to zero.")


# # plot loss
plt.figure()
plt.plot(loss_hst)
plt.savefig('trace_bottom_up_pass.png')
print(loss_hst)
 