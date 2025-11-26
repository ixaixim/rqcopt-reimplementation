import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.cnot_decompose.cnot_circuit_builder import cnot_absorb_1q_gates
from rqcopt_mpo.optimization.cnot_optimizer.cnot_optimizer import optimize
# from rqcopt_mpo.optimization.cnot_optimizer.utils import 

# trotterization params
n_sites = 6 # choose even number
J = 1.0
D = 1.5
h = 0
t = 0.25 # time of evolution

reps = 5 # debug
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
target_mpo = circuit_to_mpo(target_circ)

reps = 3 # debug
order = 1
dt = t/reps

initial_circuit = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Initial circuit with {initial_circuit.num_layers} layers")

new_circ = cnot_absorb_1q_gates(initial_circuit) # matrices are grouped 
new_circ.print_gates()

# Optimization parameters (mirroring other experiments setup)
max_steps = 20
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
max_bondim_env = 128
svd_cutoff = 0.0

loss = optimize(
    new_circ,
    target_mpo,
    max_steps=max_steps,
    max_bondim_env=max_bondim_env,
    svd_cutoff=svd_cutoff,
    lr=lr,
    betas=betas,
    eps=eps,
    clip_grad_norm=clip_grad_norm,
)
