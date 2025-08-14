# compare the weyl circuit SVD optimization loss with regular circuit optimization loss.
import rqcopt_mpo.jax_config

from pathlib import Path

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
# TODO: import optimizer and loss
from rqcopt_mpo.optimization.riemannian_adam.optimizer import optimize
import matplotlib.pyplot as plt
from experiments.utils import save_data_npz
import jax.numpy as jnp
import numpy as np

# params for initial circuit and for target
J, Delta, h = 1.0, 1.0, 0.0     # Heisenberg parameters 
n_sites = 4               # size of the chain
dt = 0.2
reps = 5
dtype = jnp.complex128
normalize_target = True
# TODO: check if Trotterized circuit with magnetic field is correctly implemented

# optimization params
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
max_steps = 20
max_bondim_env = 128
svd_cutoff = 0.0

# set up target MPO from benchmark Trotter circuit
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=Delta, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)
target_circ.print_gates()
target_mpo = circuit_to_mpo(target_circ)

if normalize_target: 
    target_mpo.normalize()
target_mpo.left_canonicalize()

# set up initial initial brickwall circuit
init_circ = trotterized_heisenberg_circuit(
    n_sites=n_sites,
    J=J,
    D=Delta,
    dt=dt,
    reps=reps,
    order=2,
    dtype=dtype,
)
# init_circ.print_gates()
# init_circ = target_circ.copy()
#run optimization
overlap = np.trace(target_mpo.dagger().to_matrix() @ init_circ.to_matrix()) 
loss = 1 - 1/2**(2*n_sites) * np.abs(overlap)**2
print(f"initial loss {loss}")
print(f"Debug: initial overlap: {overlap}")

print("Optimizing Benchmark Trotter Circuit")




# optimize circuit
loss = optimize(init_circ, target_mpo,
         lr=lr, betas=betas, eps=eps,
         clip_grad_norm=clip_grad_norm, 
         max_steps=max_steps,
         max_bondim_env=max_bondim_env,
         svd_cutoff=svd_cutoff)

# save loss data for plotting
base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, 'loss_riemannian', loss, method='Riemannian_Adam_trotterized_init')


