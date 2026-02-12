import rqcopt_mpo.jax_config

"""
Compare Trotterised time-evolution operators with the exact evolution operator
for the 6-site XYZ chain and visualise the scaling of the error
vs. the time-step Δt, using the hardware-friendly decomposition.
MPO version.
"""

import pathlib
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import (
    trotterized_hardware_friendly_xyz_circuit
)
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.tensor_network.core_ops_helpers import get_mpo_from_matrix, hs_inner_product_from_mpo

# ------------------------- problem setup ---------------------------------- #
Jx, Jy, Jz = 0.0, 0.0, 1.0
hx, hy, hz = 0.75, 0.0, 0.6
n_sites      = 6
total_time_t = 2.0

plot_order = 'all' # Now testing all orders: 1, 2, and 4


# Build XYZ Hamiltonian
model        = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                        hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
H_full       = model.build_hamiltonian_matrix()

# Exact evolution operator U(t) = exp(-i t H)
U_exact = expm(-1j * total_time_t * H_full)

# Convert exact unitary to MPO
mpo_exact = get_mpo_from_matrix(U_exact, max_bondim=128)
# Left canonicalize to ensure proper forms if needed, though get_mpo produces a valid MPO.
mpo_exact.left_canonicalize(normalize=False)

# ---------------------- helpers for Trotter circuits --------------------- #
def trotter_mpo(order: int, method: str, delta_t: float, n_steps: int) -> object:
    circ = trotterized_hardware_friendly_xyz_circuit(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order, # Pass order explicitly
        method=method,
        dt=delta_t, reps=n_steps,
        collapse=True # Use collapse for efficient MPO construction
    )
    mpo = circuit_to_mpo(circ, max_bondim=128, svd_cutoff=1e-12)
    mpo.left_canonicalize(normalize=False)
    return mpo

def frobenius_error(mpo_target, mpo_approx) -> float:
    # ||A - B||_F^2 = Tr((A-B)^dag (A-B)) = Tr(A^dag A) + Tr(B^dag B) - 2 Re(Tr(A^dag B))
    # For unitaries, Tr(A^dag A) = Tr(I) = 2^N.
    # However, MPOs might not be perfectly unitary due to truncation, so we compute all terms.
    
    # Actually, let's assume they are close to unitary or just compute properly.
    norm_target_sq = jnp.real(hs_inner_product_from_mpo(mpo_target, mpo_target))
    norm_approx_sq = jnp.real(hs_inner_product_from_mpo(mpo_approx, mpo_approx))
    overlap = hs_inner_product_from_mpo(mpo_target, mpo_approx)
    
    dist_sq = norm_target_sq + norm_approx_sq - 2 * jnp.real(overlap)
    # Avoid negative due to numerical noise
    dist_sq = jnp.maximum(dist_sq, 0.0)
    return float(jnp.sqrt(dist_sq))

def test_collapse_option_mpo():
    print("Testing 'collapse' option (MPO check)...")
    n = 6
    dt = 0.1
    reps = 2
    order = 4
    
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
    
    # MPO check
    mpo_exp = circuit_to_mpo(circ_expanded, max_bondim=128, svd_cutoff=0.0)
    mpo_col = circuit_to_mpo(circ_collapsed, max_bondim=128, svd_cutoff=0.0)
    
    # For small n, comparing via matrix is more numerically stable than overlap-based Frobenius norm
    U_exp = mpo_exp.to_matrix()
    U_col = mpo_col.to_matrix()
    diff = float(jnp.linalg.norm(U_exp - U_col))
    print(f"  Difference (Frobenius norm, matrix-based): {diff:.2e}")
    
    if diff > 1e-12:
        raise ValueError("Collapsed circuit MPO differs significantly from expanded one!")
    print("  Collapse test passed.")


# --------------------------- sweep over Δt ------------------------------- #
test_collapse_option_mpo() # Run the test

print("Starting Hardware-Friendly XYZ Trotter test (Orders 1, 2, 4) - MPO Version...")

if plot_order == 'all':
    orders_to_run = [(1, 'yoshida'), (2, 'yoshida'), (4, 'yoshida'), (4, 'suzuki')]
elif plot_order in [1, 2, 4]:
    orders_to_run = [(plot_order, 'yoshida')]
else:
    raise ValueError(f"Invalid plot_order: '{plot_order}'. Must be 1, 2, 4, or 'all'.")

n_steps_grid = np.array(np.round(np.logspace(1, 2, 10)), dtype=int)
delta_t_grid = total_time_t / n_steps_grid

errors = {om: [] for om in orders_to_run}

for i, n_steps in enumerate(n_steps_grid):
    dt = delta_t_grid[i]
    # print(f"Testing n_steps={n_steps} (Δt={dt:.4f})")
    for order, method in orders_to_run:
        mpo_tr = trotter_mpo(order=order, method=method, delta_t=dt, n_steps=n_steps)
        error = frobenius_error(mpo_exact, mpo_tr)
        errors[(order, method)].append(error)
        print(f"Order: {order:<2} | Method: {method:<8} | Steps: {n_steps:<5} | Δt: {dt:<8.4f} | Error: {error:.2e}")
# ------------------------------- plotting -------------------------------- #
plt.figure(figsize=(8, 6))
plot_styles = {
    (1, 'yoshida'): {'marker': 'o', 'linestyle': '-', 'label': '1st-order'},
    (2, 'yoshida'): {'marker': 's', 'linestyle': '-', 'label': '2nd-order (HW Friendly)'},
    (4, 'yoshida'): {'marker': '^', 'linestyle': '-', 'label': '4th-order (Yoshida)'},
    (4, 'suzuki'):  {'marker': 'v', 'linestyle': '-', 'label': '4th-order (Suzuki)'},
}

for (order, method), err_list in errors.items():
    style = plot_styles[(order, method)]
    plt.loglog(delta_t_grid, err_list, **style)
    
    # Improved ref line plotting to match the last point
    if method == 'yoshida':
        prefactor = err_list[-1] / (delta_t_grid[-1]**order)
        ref_y = prefactor * (delta_t_grid**order)
        ref_label = rf'O($\Delta t^{order}$) ref.'
        plt.loglog(delta_t_grid, ref_y, '--', label=ref_label)

plt.xlabel(r'$\Delta t$')
plt.ylabel(r'$ \| U_{\mathrm{exact}} - U_{\mathrm{Trotter}} \|_F $')
plt.title(f'Hardware-Friendly Trotter error scaling (MPO, XYZ, N={n_sites}, t={total_time_t})')
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

out_dir = pathlib.Path("plots")
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "trotter_hardware_friendly_xyz_error_mpo.png", dpi=300)
print("✓ Plot saved to", out_dir / "trotter_hardware_friendly_xyz_error_mpo.png")
