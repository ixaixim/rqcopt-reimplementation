"""
Compare Trotterised time-evolution operators with the exact evolution operator
for the 6-site XYZ chain and visualise the scaling of the error
vs. the time-step Δt.
"""

import pathlib
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
    trotterized_xyz_layers
)
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


# ------------------------- problem setup ---------------------------------- #
Jx, Jy, Jz = 1.0, 0.8, 0.5
hx, hy, hz = 0.1, 0.2, 0.3
n_sites      = 6
total_time_t = 1.0

plot_order = 'all'

# Build XYZ Hamiltonian
model        = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                        hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
H_full       = model.build_hamiltonian_matrix()

# Exact evolution operator U(t) = exp(-i t H)
U_exact = expm(-1j * total_time_t * H_full)

# ---------------------- helpers for Trotter circuits --------------------- #
def trotter_unitary(order: int, delta_t: float, n_steps: int) -> jnp.ndarray:
    layers = trotterized_xyz_layers(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order, dt=delta_t, reps=n_steps,
        dtype=jnp.complex128
    )
    circ = Circuit(
        n_sites=n_sites,
        layers=layers,
    )
    return circ.to_matrix()

def spectral_error(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(U_target - U_approx), ord=2))

# --------------------------- sweep over Δt ------------------------------- #
print("Starting XYZ Trotter test...")

if plot_order == 'all':
    orders_to_run = [1, 2, 4]
elif plot_order in [1, 2, 4]:
    orders_to_run = [plot_order]
else:
    raise ValueError(f"Invalid plot_order: '{plot_order}'. Must be 1, 2, 4, or 'all'.")

n_steps_grid = np.array(np.round(np.logspace(1, 2, 10)), dtype=int)
delta_t_grid = total_time_t / n_steps_grid

errors = {order: [] for order in orders_to_run}

for i, n_steps in enumerate(n_steps_grid):
    dt = delta_t_grid[i]
    # print(f"Testing n_steps={n_steps} (Δt={dt:.4f})")
    for order in orders_to_run:
        U_trotter = trotter_unitary(order=order, delta_t=dt, n_steps=n_steps)
        error = spectral_error(U_exact, U_trotter)
        errors[order].append(error)

# ------------------------------- plotting -------------------------------- #
plt.figure(figsize=(8, 6))
plot_styles = {
    1: {'marker': 'o', 'linestyle': '-', 'label': '1st-order'},
    2: {'marker': 's', 'linestyle': '-', 'label': '2nd-order'},
    4: {'marker': '^', 'linestyle': '-', 'label': '4th-order'},
}

for order, err_list in errors.items():
    style = plot_styles[order]
    plt.loglog(delta_t_grid, err_list, **style)
    prefactor = err_list[-1] / (delta_t_grid[-1]**order)
    ref_y = prefactor * (delta_t_grid**order)
    ref_label = rf'O($\Delta t^{order}$) ref.'
    plt.loglog(delta_t_grid, ref_y, '--', label=ref_label)

plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$ \| U_{\mathrm{exact}} - U_{\mathrm{Trotter}} \|_2 $')
plt.title(f'Trotter error scaling (XYZ, N={n_sites}, t={total_time_t})')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

out_dir = pathlib.Path("plots")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "trotter_xyz_error.png", dpi=300)
print("✓ Plot saved to", out_dir / "trotter_xyz_error.png")
