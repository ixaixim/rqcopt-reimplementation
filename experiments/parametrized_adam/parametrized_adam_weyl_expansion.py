import rqcopt_mpo.jax_config
from pathlib import Path
from experiments.utils import save_data_npz

import jax 
import jax.numpy as jnp
import numpy as np

# circuit and MPO:
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo

# decomposition:
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit
from rqcopt_mpo.circuit.decompose.single_q_decompose import euler_zyz_decompose_circuit 

# optimization: 
from rqcopt_mpo.optimization.parametrized_adam_weyl_expansion.optimizer import optimize

# trotterization:
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit

# debug
from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.optimization.utils import overlap_to_loss

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

# # set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")

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

print(f"Initial circuit with {init_circ.num_layers} layers")
print(f"Initial Fidelity of Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# optimization parameters 
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
bias_correction = True
max_steps = 1000
max_bondim_env = 128
svd_cutoff = 0.0

patience = 10
min_delta = 1e-6
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


loss = optimize(init_circ, target_mpo,
        lr=lr, betas=betas, eps=eps,
        clip_grad_norm=clip_grad_norm, 
        bias_correction=bias_correction,
        max_steps=max_steps,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff,
        callback=early_stop)

base_dir = Path(__file__).resolve().parent
save_data_npz(base_dir, 'loss_parametrized_adam_weyl', loss, method='Parametrized_Adam_Weyl')
