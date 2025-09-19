# compare the weyl circuit SVD optimization loss with regular circuit optimization loss.
import rqcopt_mpo.jax_config

from pathlib import Path

from rqcopt_mpo.circuit.circuit_builder import generate_random_brickwall_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
from rqcopt_mpo.optimization.weyl_optimizer.weyl_optimizer import optimize_weyl_circuit_local_svd
from rqcopt_mpo.optimization.utils import global_loss
import matplotlib.pyplot as plt
from experiments.utils import save_data_npz
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.optimization.utils import overlap_to_loss



# optimization params
num_sweeps = 5
layer_update_passes = 1
max_bondim_env = 256
svd_cutoff = 0.0


### Trotter initialization
# trotterization params
n_sites = 8 # choose even number
J = 1.0
D = -1.0
h = 0
t = 0.5 # time of evolution

reps = 10
order = 4
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False

# set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)

target_mpo = circuit_to_mpo(target_circ)
target_mpo.left_canonicalize(normalize=target_is_normalized)

# set up quantum circuit

reps = 3
dt = t/reps
init_circ = trotterized_heisenberg_circuit(
    n_sites=n_sites,
    J=J,
    D=D,
    dt=dt,
    reps=reps,
    order=2,
    dtype=jnp.complex128,
)

### SMALL CIRCUIT INITIALIZATION
# set up target MPO
# target_circ = trotterized_heisenberg_circuit(    
#     n_sites=n_sites, J=J, D=Delta, h=h,
#     order=4, dt=dt, reps=reps,
#     dtype=dtype
# )

# set up initial vanilla circuit
# init_circ = trotterized_heisenberg_circuit(
#     n_sites=n_sites,
#     J=J,
#     D=Delta,
#     dt=dt,
#     reps=reps,
#     order=2,
#     dtype=jnp.complex128,
# )

# init_circ = generate_random_brickwall_circuit(
#     n_sites=n_sites,    
#     n_layers=5,
#     seed=42,
#     two_qubit_name="U2",
#     dtype=dtype,
# )

# init_circ.print_gates()

# set up initial Weyl circuit
init_circ_weyl = init_circ.copy()
init_circ_weyl = weyl_decompose_circuit(init_circ_weyl)

trace = np.trace(target_circ.to_matrix().conjugate().T @ init_circ.to_matrix())
hst_cost = overlap_to_loss(trace, n_sites=n_sites, normalize=False)
print(f"Initial HST fidelity: {hst_cost}")

print("Optimizing vanilla circuit")
# optimize initial vanilla circuit
_, loss_vanilla = optimize_circuit_local_svd(
    circuit_initial=init_circ, mpo_ref=target_mpo, 
    num_sweeps=num_sweeps, layer_update_passes=layer_update_passes, 
    max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff,
    target_is_normalized=target_is_normalized
    )

print("\nOptimizing Weyl circuit")
# optimize weyl circuit
_, loss_weyl = optimize_weyl_circuit_local_svd(
    circuit_initial=init_circ_weyl, mpo_ref=target_mpo, 
    num_sweeps=num_sweeps, layer_update_passes=layer_update_passes, 
    max_bondim_env=max_bondim_env, svd_cutoff=svd_cutoff, 
    target_is_normalized=target_is_normalized
    )

num_gates_vanilla_circ = init_circ.num_gates
num_gates_weyl_circ = init_circ_weyl.num_gates

loss_vanilla = global_loss(loss_vanilla, n_sites, target_is_normalized)
loss_weyl = global_loss(loss_weyl, n_sites, target_is_normalized)

# save loss data
base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, 'loss_vanilla_circ', loss_vanilla, num_gates_vanilla_circ)
save_data_npz(base_dir, 'loss_weyl_circ', loss_weyl, num_gates_weyl_circ )


