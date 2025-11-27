"""RUN: python -m experiments.weyl_optimizer_comparison.weyl_optimizer_absorption"""
import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
from rqcopt_mpo.optimization.weyl_optimizer.weyl_abs_optimizer import optimize
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from experiments.utils import save_data_npz

# parameters: model
n_sites = 10 # choose even number
J = 1.0
D = 1.5
h = 0
t = 0.25 # time of evolution

reps = 10 #debug 
order = 4
dt = t/reps
dtype = jnp.complex128
normalize = False

# parameters: optimization
num_sweeps = 1500 # debug
max_bondim_env = 128
svd_cutoff = 0.0

lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
bias_correction = True

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


# set up target MPO
print(f"Setting up Reference MPO from Trotter Circuit of order {order}, with {reps} repetitions...")
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)

target_mpo = circuit_to_mpo(target_circ)
target_mpo.left_canonicalize(normalize=normalize)

# set up initial circuit
reps = 3
dt = t/reps
order = 2
circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=order, dt=dt, reps=reps,
    dtype=dtype
)

# set up circuit: absorb gates for parameter reduction
weyl_circ = weyl_decompose_circuit(circ, keep_global_phase=False)
weyl_circ = absorb_single_qubit_layers(weyl_circ)
weyl_circ.print_gates()


# optimize
circ, loss = optimize(
    weyl_circ, 
    mpo_ref=target_mpo, 
    lr=lr, betas=betas, eps=eps,
    clip_grad_norm=clip_grad_norm, 
    bias_correction=bias_correction,
    max_steps=num_sweeps,
    max_bondim_env=max_bondim_env,
    svd_cutoff=svd_cutoff,
    callback=early_stop,
    )

base_dir = Path(__file__).resolve().parent
save_data_npz(base_dir, f"loss_weyl_absorption_reps_{reps}", loss, method="Weyl_Absorption")
