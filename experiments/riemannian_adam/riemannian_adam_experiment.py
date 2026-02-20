import rqcopt_mpo.jax_config

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_xyz_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.riemannian_adam.optimizer import optimize
from experiments.utils import save_data_npz
from rqcopt_mpo.optimization.utils import overlap_to_loss


# trotterization params
n_sites = 6 # choose even number
J = 0. # HAS TO BE ZERO
D = 1.
hx = 0.75
hz = 0.6
t = 2.0 # time of evolution

reps = 20 # debug
order = 4
dt = t/reps
dtype = jnp.complex128
target_is_normalized = True

max_bondim_ref = 64


# set up target MPO

# target_circ = trotterized_hardware_friendly_xyz_circuit(    
#     n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz,
#     order=order, dt=dt, reps=reps,
#     dtype=dtype
# )

target_circ = trotterized_xyz_circuit(
    n_sites=n_sites,
    Jx=J, Jy=J, Jz=D,
    hx=hx, hz=hz,
    order=order,
    method='suzuki',
    dt=dt,
    reps=reps,
    dtype=dtype
)

print(f"Target circuit with {target_circ.num_2q_layers} layers")

target_mpo = circuit_to_mpo(target_circ, max_bondim=max_bondim_ref, svd_cutoff=0.0)
target_mpo.left_canonicalize(normalize=target_is_normalized)

# set up quantum circuit

reps = 5
dt = t/reps
order = 2

# init_circ = trotterized_hardware_friendly_xyz_circuit(
#     n_sites=n_sites,
#     Jx=J,
#     Jy=J,
#     Jz=D,
#     hx=hx,
#     hz=hz,
#     dt=dt,
#     reps=reps,
#     order=order,
#     dtype=jnp.complex128,
# )

init_circ = trotterized_xyz_circuit(
    n_sites=n_sites,
    Jx=J,
    Jy=J,
    Jz=D,
    hx=hx,
    hz=hz,
    dt=dt,
    reps=reps,
    order=order,
    method='suzuki',
    dtype=jnp.complex128,
)

print(f"Initial circuit with {init_circ.num_2q_layers} two-qubit layers")
# print(f"Initial Fidelity of Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# # weyl decomp
# init_circ = weyl_decompose_circuit(init_circ, keep_global_phase=False)
# # euler rotation decomp
# init_circ = euler_zyz_decompose_circuit(init_circ, include_global_phase=False)
# print(f"Init Circuit Decomposed with {init_circ.num_layers} layers")
# print(f"Initial Fidelity of Decomposed Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# Optimization parameters
max_steps = 2000 #debug
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
bias_correction = True
max_bondim_env = 128
svd_cutoff = 0.0

print("Optimizing brickwall circuit (Riemannian Adam)")

patience = 10
min_delta = 1e-9
best_loss = [np.inf]
stalled_steps = [0]


def early_stop(*, step: int, loss: float, **_):
    if loss < best_loss[0] - min_delta:
        best_loss[0] = loss
        stalled_steps[0] = 0
    else:
        stalled_steps[0] += 1

    if stalled_steps[0] >= patience:
        print(f"Early stopping at step {step}")
        return True
    return False


# optimize circuit
circ, loss = optimize(
    init_circ,
    target_mpo,
    lr=lr,
    betas=betas,
    eps=eps,
    clip_grad_norm=clip_grad_norm,
    max_steps=max_steps,
    max_bondim_env=max_bondim_env,
    svd_cutoff=svd_cutoff,
    callback=early_stop,
)

# save loss data for plotting
lr_tag = f"{lr:.0e}".replace(".", "p")
base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, f'loss_riemannian_sites_{n_sites}_reps_{reps}_lr_{lr_tag}', loss, method='Riemannian_Adam')
