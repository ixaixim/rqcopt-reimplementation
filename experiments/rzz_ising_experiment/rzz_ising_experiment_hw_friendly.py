import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

# MPO builder and circuits
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import rzz_decompose_ising_circuit #hw friendly
from rqcopt_mpo.optimization.rzz_ising_optimizer.rzz_ising_optimizer_hw_friendly import optimize # hw_friendly
# from rqcopt_mpo.optimization.utils import overlap_to_loss
# from rqcopt_mpo.optimization.adam_utils import make_early_stop
# from rqcopt_mpo.optimization.cnot_optimizer.utils import 
# from experiments.utils import save_data_npz


n_sites = 6 # choose even number
J = 0. # HAS TO BE ZERO
D = 1.
hx, hz = 0.75, 0.25
t = 0.25 # time of evolution

if J != 0 or hx==0 or hz==0:
    raise ValueError("Only Transverse Field Ising Model (TFIM) is permitted here.")

reps = 10 
order = 4 
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False 

max_steps = 1000
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
max_bondim_env = 128
svd_cutoff = 0.0

# patience = 10
# min_delta = 1e-8
# early_stop = make_early_stop(patience=patience, min_delta=min_delta)



target_circ = trotterized_hardware_friendly_xyz_circuit(    
    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz,
    order=order, dt=dt, reps=reps, collapse=True,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")
target_mpo = circuit_to_mpo(target_circ)

reps = 2
dt = t/reps
order = 2

initial_circuit = trotterized_hardware_friendly_xyz_circuit(    
    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz, 
    order=order, dt=dt, reps=reps, collapse=True,
    dtype=dtype
)
# initial_circuit.print_gates()
# print(f"Initial circuit with {initial_circuit.num_layers} layers")

# decompose and parameterize circuit.
new_circ = rzz_decompose_ising_circuit(initial_circuit, order) # matrices are grouped and parametrized 
print(f"New circuit with {new_circ.num_layers} layers")
# new_circ.print_gates()

circ, loss = optimize(
    new_circ,
    target_mpo,
    max_steps=max_steps,
    max_bondim_env=max_bondim_env,
    svd_cutoff=svd_cutoff,
    lr=lr,
    betas=betas,
    eps=eps,
    clip_grad_norm=clip_grad_norm,
    # callback=early_stop, 
)


