# compare the weyl circuit SVD optimization loss with regular circuit optimization loss.
import rqcopt_mpo.jax_config

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from rqcopt_mpo.circuit.circuit_builder import generate_random_brickwall_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit, absorb_single_qubit_layers
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
from rqcopt_mpo.optimization.weyl_optimizer.weyl_optimizer import optimize_weyl_circuit_local_svd
from rqcopt_mpo.optimization.utils import global_loss
import matplotlib.pyplot as plt
from experiments.utils import save_data_npz
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.optimization.utils import overlap_to_loss



# optimization params
num_sweeps = 20
layer_update_passes = 1
max_bondim_env = 128
svd_cutoff = 0.0

# early stopping params
early_stop_patience = 10
early_stop_tol = 1e-6
early_stop_target: Optional[float] = None


def run_optimizer_with_early_stop(
    optimize_fn: Callable[..., Tuple],
    circuit_initial,
    *,
    mpo_ref,
    n_sites: int,
    max_sweeps: int,
    max_bondim_env: int,
    layer_update_passes: int,
    svd_cutoff: float,
    target_is_normalized: bool,
    early_stop_patience: int,
    early_stop_tol: float,
    early_stop_target: Optional[float] = None,
) -> Tuple[object, List[complex], int]:
    """Run a circuit optimizer sweep-by-sweep with early stopping."""

    circuit = circuit_initial
    aggregated_loss: List[complex] = []
    best_loss = np.inf
    sweeps_without_improvement = 0
    sweeps_completed = 0

    for sweep_idx in range(max_sweeps):
        circuit, sweep_loss_history = optimize_fn(
            circuit_initial=circuit,
            mpo_ref=mpo_ref,
            num_sweeps=1,
            max_bondim_env=max_bondim_env,
            layer_update_passes=layer_update_passes,
            svd_cutoff=svd_cutoff,
            target_is_normalized=target_is_normalized,
        )

        sweeps_completed = sweep_idx + 1
        aggregated_loss.extend(sweep_loss_history)

        if not sweep_loss_history:
            print("No loss history returned; stopping optimization early.")
            break

        sweep_losses = global_loss(
            sweep_loss_history, n_sites=n_sites, normalize=target_is_normalized
        )
        sweep_losses_np = np.asarray(sweep_losses, dtype=np.float64)
        current_loss = float(sweep_losses_np[-1])

        if early_stop_target is not None and current_loss <= early_stop_target:
            print(
                f"Early stopping: reached target loss {current_loss:.3e} "
                f"at sweep {sweeps_completed}."
            )
            break

        if best_loss - current_loss > early_stop_tol:
            best_loss = current_loss
            sweeps_without_improvement = 0
        else:
            sweeps_without_improvement += 1
            if sweeps_without_improvement >= early_stop_patience:
                print(
                    f"Early stopping: no improvement greater than {early_stop_tol} "
                    f"for {early_stop_patience} sweep(s)."
                )
                break
    else:
        print(f"Reached max sweeps ({max_sweeps}) without triggering early stop.")

    return circuit, aggregated_loss, sweeps_completed


### Trotter initialization
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
target_is_normalized = True

# set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)

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

# set up initial Weyl circuit
# Add a further comparison with the Weyl-absorbed circuit.
init_circ_weyl = init_circ.copy()
init_circ_weyl = weyl_decompose_circuit(init_circ_weyl)
init_circ_weyl_abs = absorb_single_qubit_layers(init_circ_weyl)

trace = np.trace(target_circ.to_matrix().conjugate().T @ init_circ.to_matrix())
hst_cost = overlap_to_loss(trace, n_sites=n_sites, normalize=False)
print(f"Initial HST fidelity: {hst_cost}")

print("Optimizing vanilla circuit")
# optimize initial vanilla circuit
_, loss_vanilla_raw, sweeps_vanilla = run_optimizer_with_early_stop(
    optimize_circuit_local_svd,
    init_circ,
    mpo_ref=target_mpo,
    n_sites=n_sites,
    max_sweeps=num_sweeps,
    max_bondim_env=max_bondim_env,
    layer_update_passes=layer_update_passes,
    svd_cutoff=svd_cutoff,
    target_is_normalized=target_is_normalized,
    early_stop_patience=early_stop_patience,
    early_stop_tol=early_stop_tol,
    early_stop_target=early_stop_target,
)
print(f"Vanilla circuit optimization completed {sweeps_vanilla} sweep(s).")

print("\nOptimizing Weyl circuit")
# optimize weyl circuit
_, loss_weyl_raw, sweeps_weyl = run_optimizer_with_early_stop(
    optimize_weyl_circuit_local_svd,
    init_circ_weyl,
    mpo_ref=target_mpo,
    n_sites=n_sites,
    max_sweeps=num_sweeps,
    max_bondim_env=max_bondim_env,
    layer_update_passes=layer_update_passes,
    svd_cutoff=svd_cutoff,
    target_is_normalized=target_is_normalized,
    early_stop_patience=early_stop_patience,
    early_stop_tol=early_stop_tol,
    early_stop_target=early_stop_target,
)
print(f"Weyl circuit optimization completed {sweeps_weyl} sweep(s).")

print("\nOptimizing Weyl-absorbed circuit")
# optimize weyl-absorbed circuit
_, loss_weyl_abs_raw, sweeps_weyl_abs = run_optimizer_with_early_stop(
    optimize_weyl_circuit_local_svd,
    init_circ_weyl_abs,
    mpo_ref=target_mpo,
    n_sites=n_sites,
    max_sweeps=num_sweeps,
    max_bondim_env=max_bondim_env,
    layer_update_passes=layer_update_passes,
    svd_cutoff=svd_cutoff,
    target_is_normalized=target_is_normalized,
    early_stop_patience=early_stop_patience,
    early_stop_tol=early_stop_tol,
    early_stop_target=early_stop_target,
)
print(f"Weyl-absorbed circuit optimization completed {sweeps_weyl_abs} sweep(s).")

num_gates_vanilla_circ = init_circ.num_gates
num_gates_weyl_circ = init_circ_weyl.num_gates
num_gates_weyl_abs_circ = init_circ_weyl_abs.num_gates

loss_vanilla = global_loss(loss_vanilla_raw, n_sites, target_is_normalized)
loss_weyl = global_loss(loss_weyl_raw, n_sites, target_is_normalized)
loss_weyl_abs = global_loss(loss_weyl_abs_raw, n_sites, target_is_normalized)

# save loss data
base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, 'loss_vanilla_circ', loss_vanilla, num_gates_vanilla_circ)
save_data_npz(base_dir, 'loss_weyl_circ', loss_weyl, num_gates_weyl_circ )
save_data_npz(base_dir, 'loss_weyl_abs_circ', loss_weyl_abs, num_gates_weyl_abs_circ)
