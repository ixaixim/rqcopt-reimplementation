"""
Compare Trotterised time-evolution operators with the exact evolution operator
for the 4-site Heisenberg chain and visualise the scaling of the error
vs. the time-step Δt.
"""

import pathlib
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm                    # exact evolution
import matplotlib.pyplot as plt
from rqcopt_mpo.hamiltonian.heisenberg import HeisenbergModel
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
    trotterized_heisenberg_layers
)
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


# ------------------------- problem setup ---------------------------------- #
J, Delta, h = 1.0, 1.0, 1.0     # Heisenberg parameters (XXX model)
n_sites      = 4               # size of the chain
total_time_t = 1.0              # simulate up to t = 1

# --- MODIFICATION START: Control which Trotter order(s) to plot ---
# Set this variable to 1, 2, 4, or 'all' to control the output.
plot_order = 'all'
# --- MODIFICATION END ---


# Build Heisenberg Hamiltonian and collect local terms (needed only for ref-plot).
model        = HeisenbergModel(n_sites=n_sites, J=J, Delta=Delta, h=h,
                               dtype=jnp.complex128)
H_full       = model.build_hamiltonian_matrix()            # 2ᴺ×2ᴺ dense matrix (N=4 → 16×16)

# Exact evolution operator U(t) = exp(-i t H)
U_exact = expm(-1j * total_time_t * H_full)

# ---------------------- helpers for Trotter circuits --------------------- #
def trotter_unitary(order: int, delta_t: float, n_steps: int) -> jnp.ndarray:
    """
    Build an `order`-th-order Suzuki-Trotter circuit and return its dense unitary
    matrix for a given Δt and number of steps n such that n * Δt = t.
    """
    layers = trotterized_heisenberg_layers(
        n_sites=n_sites, J=J, D=Delta, h=h,
        order=order, dt=delta_t, reps=n_steps,
        dtype=jnp.complex128
    )
    circ = Circuit(
        n_sites=n_sites,
        layers=layers,
    )
    return circ.to_matrix()


def spectral_error(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
    """‖U_target − U_approx‖₂ (spectral / operator norm)."""
    return float(np.linalg.norm(np.asarray(U_target - U_approx), ord=2))


# --------------------------- sweep over Δt ------------------------------- #
print("Starting program...")

# --- MODIFICATION START: Determine which orders to run based on the control variable ---
if plot_order == 'all':
    orders_to_run = [1, 2, 4]
elif plot_order in [1, 2, 4]:
    orders_to_run = [plot_order]
else:
    raise ValueError(f"Invalid plot_order: '{plot_order}'. Must be 1, 2, 4, or 'all'.")
print(f"Will calculate and plot for order(s): {orders_to_run}")
# --- MODIFICATION END ---


n_steps_grid = np.array(np.round(np.logspace(1, 2, 10)), dtype=int)  # e.g., [10, 13, 17, ..., 1000]
delta_t_grid = total_time_t / n_steps_grid  # Calculate dt to make total time exact


# --- MODIFICATION START: Use a dictionary to store errors for flexibility ---
# The keys are the Trotter orders and values are lists of errors.
errors = {order: [] for order in orders_to_run}
# --- MODIFICATION END ---

for i, n_steps in enumerate(n_steps_grid):
    dt = delta_t_grid[i]
    print(f"Testing n_steps={n_steps} (Δt={dt:.4f})")

    # --- MODIFICATION START: Loop over the selected orders to calculate errors ---
    for order in orders_to_run:
        U_trotter = trotter_unitary(order=order, delta_t=dt, n_steps=n_steps)
        error = spectral_error(U_exact, U_trotter)
        errors[order].append(error)
    # --- MODIFICATION END ---


# ------------------------------- plotting -------------------------------- #
plt.figure(figsize=(8, 6))

# --- MODIFICATION START: Use dictionaries for plot styles for cleaner code ---
plot_styles = {
    1: {'marker': 'o', 'linestyle': '-', 'label': '1st-order'},
    2: {'marker': 's', 'linestyle': '-', 'label': '2nd-order'},
    4: {'marker': '^', 'linestyle': '-', 'label': '4th-order'},
}
# --- MODIFICATION END ---


# --- MODIFICATION START: Loop through the results to plot data and reference lines ---
for order, err_list in errors.items():
    # Plot the calculated error data
    style = plot_styles[order]
    plt.loglog(delta_t_grid, err_list, **style)

    # Plot the corresponding reference slope O(Δt^order)
    # Choose a prefactor so the line goes through that data point
    # prefactor = err_list[0] / (delta_t_grid[0]**order)
    prefactor = err_list[-1] / (delta_t_grid[-1]**order) # New way

    ref_y = prefactor * (delta_t_grid**order)
    ref_label = rf'O($\Delta t^{order}$) ref.'
    plt.loglog(delta_t_grid, ref_y, '--', label=ref_label)
# --- MODIFICATION END ---


plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$ \| U_{\mathrm{exact}} - U_{\mathrm{Trotter}} \|_2 $')
plt.title(f'Trotter error scaling (Heisenberg, N={n_sites}, t={total_time_t})')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5) # Added grid for readability
plt.tight_layout()

out_dir = pathlib.Path("plots")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "trotter_error.png", dpi=300)
print("✓ Plot saved to", out_dir / "trotter_error.png")