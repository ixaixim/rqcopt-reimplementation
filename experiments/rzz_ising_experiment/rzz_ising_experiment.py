import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

# MPO builder and circuits
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit, trotterized_xyz_circuit
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import rzz_decompose_ising_circuit
from rqcopt_mpo.optimization.rzz_ising_optimizer.rzz_ising_optimizer import optimize
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.optimization.adam_utils import make_early_stop
# from rqcopt_mpo.optimization.cnot_optimizer.utils import 
from experiments.utils import save_data_npz

# trotterization params
n_sites = 6 # choose even number
J = 0. # HAS TO BE ZERO
D = 0.5
h = 1.2
t = 1. # time of evolution
if J != 0 or h==0:
    raise ValueError("Only Transverse Field Ising Model (TFIM) is permitted here.")

reps = 4 #debug 
order = 4
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False

# set up target MPO
# target_circ = trotterized_heisenberg_circuit(    
#     n_sites=n_sites, J=J, D=D, h=h,
#     order=order, dt=dt, reps=reps,
#     dtype=dtype
# )
target_circ = trotterized_xyz_circuit(    
    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")
target_mpo = circuit_to_mpo(target_circ)

reps = 3
dt = t/reps
order = 2

# initial_circuit = trotterized_heisenberg_circuit(    
#     n_sites=n_sites, J=J, D=D, h=h,
#     order=order, dt=dt, reps=reps,
#     dtype=dtype
# )
initial_circuit = trotterized_xyz_circuit(    
    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Initial circuit with {initial_circuit.num_layers} layers")

# trace = np.trace(
#     target_circ.to_matrix().conjugate().T @ initial_circuit.to_matrix()
# )
# initial_loss = overlap_to_loss(trace, n_sites=n_sites, normalize=False)
# print(f"Initial loss (HST cost) without optimization: {initial_loss}")

new_circ = rzz_decompose_ising_circuit(initial_circuit) # matrices are grouped 
# new_circ.print_gates()

# Optimization parameters (mirroring other experiments setup)
max_steps = 1000
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
max_bondim_env = 128
svd_cutoff = 0.0

patience = 10
min_delta = 1e-8
early_stop = make_early_stop(patience=patience, min_delta=min_delta)
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
    callback=early_stop
)

# lr_tag = f"{lr:.0e}".replace(".", "p")
# base_dir = Path(__file__).resolve().parent
# save_data_npz(base_dir, f"loss_cnot_reps_{reps}_lr_{lr_tag}", loss, method="cnot_block")
