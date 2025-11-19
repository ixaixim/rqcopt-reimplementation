import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.cnot_decompose.cnot_circuit_builder import cnot_absorb_1q_gates

# trotterization params
n_sites = 4 # choose even number
J = 1.0
D = 1.5
h = 0
t = 0.25 # time of evolution

reps = 1 # debug
order = 2
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False

# set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")
target_circ.print_gates()

new_circ = cnot_absorb_1q_gates(target_circ)
new_circ.print_gates()