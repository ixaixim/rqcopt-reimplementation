"""
Compare Trotterised time-evolution operators with the exact evolution operator
for the 6-site XYZ chain and visualise the scaling of the error
vs. the time-step Δt, using the hardware-friendly decomposition.
"""

import pathlib
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import (
    trotterized_hardware_friendly_xyz_layers,
    trotterized_hardware_friendly_xyz_circuit
)
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


# ------------------------- problem setup ---------------------------------- #
Jx, Jy, Jz = 1.0, 0.8, 0.5
hx, hy, hz = 0.1, 0.2, 0.3
n_sites      = 6
total_time_t = 1.0

plot_order = 'all' # Now testing all orders: 1, 2, and 4


# Build XYZ Hamiltonian
model        = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                        hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
H_full       = model.build_hamiltonian_matrix()

# Exact evolution operator U(t) = exp(-i t H)
U_exact = expm(-1j * total_time_t * H_full)

# ---------------------- helpers for Trotter circuits --------------------- #
def trotter_unitary(order: int, delta_t: float, n_steps: int) -> jnp.ndarray:
    layers = trotterized_hardware_friendly_xyz_layers(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order, # Pass order explicitly
        dt=delta_t, reps=n_steps,
        dtype=jnp.complex128
    )
    circ = Circuit(
        n_sites=n_sites,
        layers=layers,
    )
    return circ.to_matrix()

def spectral_error(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(U_target - U_approx), ord=2))

def test_collapse_option():
    print("Testing 'collapse' option...")
    n = 4
    dt = 0.1
    reps = 1
    order = 2
    
    # 1. Standard (Expanded)
    circ_expanded = trotterized_hardware_friendly_xyz_circuit(
        n_sites=n, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
        dt=dt, reps=reps, order=order, collapse=False
    )
    # 2. Collapsed
    circ_collapsed = trotterized_hardware_friendly_xyz_circuit(
        n_sites=n, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
        dt=dt, reps=reps, order=order, collapse=True
    )
    
    print(f"  Layers (Expanded): {len(circ_expanded.layers)}")
    print(f"  Layers (Collapsed): {len(circ_collapsed.layers)}")
    
    # Expect fewer layers.
    # Expanded order 2 (1 rep): E(dt/2) + [X,Y,Z](dt/2) + O(dt) + [Z,Y,X](dt/2) + E(dt/2)
    # Layers: 1 + 3 + 1 + 3 + 1 = 9
    # Collapsed: 1 + 1 + 1 + 1 + 1 = 5
    if len(circ_collapsed.layers) >= len(circ_expanded.layers):
        print("  WARNING: Layer count did not decrease as expected.")
    
    # Matrix check
    U_exp = circ_expanded.to_matrix()
    U_col = circ_collapsed.to_matrix()
    
    diff = spectral_error(U_exp, U_col)
    print(f"  Difference (spectral norm): {diff:.2e}")
    
    if diff > 1e-12:
        raise ValueError("Collapsed circuit unitary differs significantly from expanded one!")
    print("  Collapse test passed.")

# --------------------------- sweep over Δt ------------------------------- #
test_collapse_option() # Run the test

print("Starting Hardware-Friendly XYZ Trotter test (Orders 1, 2, 4)...")

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
    2: {'marker': 's', 'linestyle': '-', 'label': '2nd-order (HW Friendly)'},
    4: {'marker': '^', 'linestyle': '-', 'label': '4th-order (Yoshida)'},
}

for order, err_list in errors.items():
    style = plot_styles[order]
    plt.loglog(delta_t_grid, err_list, **style)
    
    # Improved ref line plotting to match the last point
    prefactor = err_list[-1] / (delta_t_grid[-1]**order)
    ref_y = prefactor * (delta_t_grid**order)
    ref_label = rf'O($\Delta t^{order}$) ref.'
    plt.loglog(delta_t_grid, ref_y, '--', label=ref_label)

plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$ \| U_{\mathrm{exact}} - U_{\mathrm{Trotter}} \|_2 $')
plt.title(f'Hardware-Friendly Trotter error scaling (XYZ, N={n_sites}, t={total_time_t})')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

out_dir = pathlib.Path("plots")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "trotter_hardware_friendly_xyz_error.png", dpi=300)
print("✓ Plot saved to", out_dir / "trotter_hardware_friendly_xyz_error.png")
