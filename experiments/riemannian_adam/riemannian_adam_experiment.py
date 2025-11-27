import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.riemannian_adam.optimizer import optimize
from experiments.utils import save_data_npz
from rqcopt_mpo.optimization.utils import overlap_to_loss


# trotterization params
n_sites = 10 # choose even number
J = 1.0
D = 1.5
h = 0
t = 0.25 # time of evolution

reps = 4 # debug
order = 4
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
target_mpo.left_canonicalize(normalize=target_is_normalized)

# set up quantum circuit

reps = 3
dt = t/reps
order = 2
init_circ = trotterized_heisenberg_circuit(
    n_sites=n_sites,
    J=J,
    D=D,
    dt=dt,
    reps=reps,
    order=order,
    dtype=jnp.complex128,
)

print(f"Initial circuit with {init_circ.num_layers} layers")
print(f"Initial Fidelity of Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# # weyl decomp
# init_circ = weyl_decompose_circuit(init_circ, keep_global_phase=False)
# # euler rotation decomp
# init_circ = euler_zyz_decompose_circuit(init_circ, include_global_phase=False)
# print(f"Init Circuit Decomposed with {init_circ.num_layers} layers")
# print(f"Initial Fidelity of Decomposed Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# Optimization parameters
max_steps = 20 #debug
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
bias_correction = True
max_bondim_env = 128
svd_cutoff = 0.0

print("Optimizing brickwall circuit (Riemannian Adam)")

patience = 10
min_delta = 1e-8
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
base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, f'loss_riemannian_reps_{reps}', loss, method='Riemannian_Adam')
