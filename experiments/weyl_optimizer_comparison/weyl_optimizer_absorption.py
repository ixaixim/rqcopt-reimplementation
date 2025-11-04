import rqcopt_mpo.jax_config

import jax.numpy as jnp
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
from rqcopt_mpo.optimization.weyl_optimizer.weyl_abs_optimizer import optimize
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo

# parameeters: model
n_sites = 6 # choose even number
J = 1.0
D = -1.0
h = 0
t = 0.06 # time of evolution

reps = 10
order = 4
dt = t/reps
dtype = jnp.complex128
normalize = False

# parameters: optimization
num_sweeps = 1
max_bondim_env = 128
svd_cutoff = 0.0

# set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)

target_mpo = circuit_to_mpo(target_circ)
target_mpo.left_canonicalize(normalize=normalize)

# set up initial circuit
reps = 1
dt = t/reps
circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=1, dt=dt, reps=reps,
    dtype=dtype
)

# set up circuit: absorb gates for parameter reduction
weyl_circ = weyl_decompose_circuit(circ, keep_global_phase=False)
weyl_circ = absorb_single_qubit_layers(weyl_circ)
weyl_circ.print_gates()

# optimize
loss = optimize(
    weyl_circ, 
    mpo_ref=target_mpo, 
    num_sweeps=num_sweeps,
    max_bondim_env=max_bondim_env,
    svd_cutoff=svd_cutoff
    )


