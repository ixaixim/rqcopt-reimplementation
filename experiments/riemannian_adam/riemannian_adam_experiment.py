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
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.riemannian_adam.optimizer import optimize
from experiments.utils import save_data_npz, save_experiment_json, get_reference_path
from rqcopt_mpo.optimization.utils import overlap_to_loss
from experiments.riemannian_adam.analysis import analyze_circuit_weyl_redundancy


# trotterization params
n_sites = 6 # choose even number
J = 0. # HAS TO BE ZERO
D = 1.
hx = 0.75
hz = 0.6
t = 2.0 # time of evolution

reps_ref = 20 # debug
order_ref = 4
dtype = jnp.complex128
target_is_normalized = True

# set up target MPO
ref_base_dir = REPO_ROOT / "experiments"
target_path = get_reference_path(
    base_dir=ref_base_dir,
    n_sites=n_sites,
    Jx=J, Jy=J, Jz=D,
    hx=hx, hy=0.0, hz=hz,
    t=t, reps=reps_ref, order=order_ref
)

if not target_path.exists():
    print(f"Reference MPO not found at {target_path}. Run experiments/create_reference.py first.")
    sys.exit(1)

target_mpo = MPO.load_json(target_path)
print(f"Loaded target MPO from {target_path}")

# set up quantum circuit

reps = 3
dt = t/reps
order = 2

init_circ = trotterized_xyz_circuit(
    n_sites=n_sites,
    Jx=J, Jy=J, Jz=D,
    hx=hx, hz=hz,
    dt=dt,
    reps=reps,
    order=order,
    method='suzuki',
    dtype=jnp.complex128,
)

print(f"Initial circuit with {init_circ.num_2q_layers} two-qubit layers")

# Optimization parameters
max_steps = 50 #debug
lr = 1e-3
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

# II order: l(r) = 2*r + 1 
# II order after synthesis: l(r) = 3* (2r + 1)
# save loss data for plotting
# lr_tag = f"{lr:.0e}".replace(".", "p")
# save_data_npz(base_dir, f'loss_riemannian_sites_{n_sites}_reps_{reps}_lr_{lr_tag}', loss, method='Riemannian_Adam')
base_dir = here = Path(__file__).resolve().parent

# Analyze the optimized circuit redundancy
total_physical_layers = analyze_circuit_weyl_redundancy(circ)

circuit_filename = f"circuit_riemannian_reps_{reps}.json"
circ.save_json(base_dir / "data" / circuit_filename)

save_experiment_json(
    base_dir=base_dir,
    method="riemannian-opt",
    circuit=circ,
    final_loss=loss[-1],
    hamiltonian_params={"n_sites": n_sites, "J": J, "D": D, "hx": hx, "hz": hz, "t": t},
    trotter_params={"reps_ansatz": reps, "order_ansatz": order},
    optimization_params={
        "lr": lr,
        "betas": betas,
        "eps": eps,
        "max_steps": max_steps,
        "max_bondim_env": max_bondim_env,
        "svd_cutoff": svd_cutoff,
        "patience": patience,
        "min_delta": min_delta,
    },
    loss_history=loss,
    additional_data={"total_physical_layers": total_physical_layers},
)
