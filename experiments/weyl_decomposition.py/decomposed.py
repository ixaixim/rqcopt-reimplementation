import rqcopt_mpo.jax_config

import jax.numpy as jnp
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers

n_sites = 6 # choose even number
J = 1.0
D = -1.0
h = 0
t = 0.06 # time of evolution

reps = 1
order = 4
dt = t/reps
dtype = jnp.complex128
normalize = False

# set up target MPO
circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)

# set up circuit
weyl_circ = weyl_decompose_circuit(circ, keep_global_phase=False)
weyl_circ = absorb_single_qubit_layers(weyl_circ)
weyl_circ.print_gates()



