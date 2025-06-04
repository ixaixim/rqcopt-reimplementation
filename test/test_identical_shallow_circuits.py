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

n_sites = 7
# generate random target circuit.
target_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=8,
    p_single=0.3,
    p_two=0.3,
    seed=43,
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


# generate initial circuit with same layout, and same gates.
init_circuit = target_circuit.copy()
print("Generated initial circuit with the same layout and gates as the target circuit:")
init_circuit.print_gates()

# test that the two circuits are actually the same unitary matrix: Tr(init_circuit^dag target_circuit) = Tr(target_circuit^dagger init_circuit)
# init_matrix = init_circuit.to_matrix()
# target_matrix = target_circuit.to_matrix()
init_matrix = circuit_to_mpo(init_circuit).to_matrix()
target_matrix = circuit_to_mpo(target_circuit).to_matrix()
tr1 = np.trace(init_matrix.conjugate().T @ target_matrix)
tr2 = np.trace(target_matrix.conjugate().T @ init_matrix)
print(f"Trace 1: {tr1}, Trace 2: {tr2}")


# run optimization routine. 
_, loss = optimize_circuit_local_svd(circuit_initial=init_circuit, mpo_ref=target_mpo, num_sweeps=1, max_bondim_env=128, svd_cutoff=1e-14)

loss_hst = 1 - 1/2**(2*n_sites) * np.abs(loss)**2
if jnp.allclose(jnp.array(loss_hst), 0, atol=1e-12):
    print("Result is close to zero within atol=1e-12.")
else:
    print("Result is NOT close to zero.")


# # plot loss
plt.figure()
plt.plot(loss_hst)
plt.savefig('trace_bottom_up_pass.png')
print(loss_hst)
