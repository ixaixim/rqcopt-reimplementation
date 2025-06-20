# test the compression of a trotterized circuit (4th order), using a 2nd order trotter circuit 
import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotter_circuit, compress_consecutive_layers
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

n_sites = 4 # choose even number
J = 0.0
Delta = 1.0
# initialize reference circuit
# order 4, 5 reps, dt=t/5, total time=1.
target_circuit = trotter_circuit(
    n_sites=n_sites,
    J=J,
    Delta=Delta,
    delta_t=0.2,
    n_steps=5,
    order=4,
    combine_gate_triplet=True,
    dtype=jnp.complex128,
)
compress_consecutive_layers(target_circuit)
target_circuit.print_gates()
target_mpo = circuit_to_mpo(target_circuit)
target_mpo.left_canonicalize()
norm = target_mpo.normalize()
print(f'Target norm = {norm}')

# initialize circuit 
# order 2, 2 reps,  same dt.
init_circuit = trotter_circuit(
    n_sites=n_sites,
    J=J,
    Delta=Delta,
    delta_t=0.2,
    n_steps=5,
    order=2,
    combine_gate_triplet=True,
    dtype=jnp.complex128,
)
compress_consecutive_layers(init_circuit)
init_circuit.print_gates()


_, loss = optimize_circuit_local_svd(
    circuit_initial=init_circuit, mpo_ref=target_mpo, 
    num_sweeps=10, layer_update_passes=1, 
    max_bondim_env=128, svd_cutoff=0.0,
    )

loss_hst = 1 - 1/2**(n_sites) * np.abs(loss)**2
# if jnp.allclose(jnp.array(loss_hst), 0, atol=1e-12):
#     print("Result is close to zero within atol=1e-12.")
# else:
#     print("Result is NOT close to zero.")


# plot loss
plt.figure()
plt.plot(loss_hst)
plt.savefig('loss_no_weyl.png')
print(loss_hst)


 
