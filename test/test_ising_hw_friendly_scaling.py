import pathlib
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt
from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import trotterized_ising_hw_friendly_circuit

# ------------------------- problem setup ---------------------------------- #
J = 1.0
hx, hz = 0.75, 0.6
n_sites      = 6
total_time_t = 1.0

# Build Ising Hamiltonian using XYZModel as a reference
model        = XYZModel(n_sites=n_sites, Jx=0.0, Jy=0.0, Jz=J, 
                        hx=hx, hy=0.0, hz=hz, dtype=jnp.complex128)
H_full       = model.build_hamiltonian_matrix()

# Exact evolution operator U(t) = exp(-i t H)
U_exact = expm(-1j * total_time_t * H_full)

def spectral_error(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(U_target - U_approx), ord=2))

def test_ising_trotter_scaling():
    print(f"Testing Ising Hardware-Friendly Trotter scaling (N={n_sites}, t={total_time_t})...")
    
    orders = [1, 2, 4]
    n_steps_grid = np.array(np.round(np.logspace(0.8, 2.0, 8)), dtype=int)
    delta_t_grid = total_time_t / n_steps_grid
    
    errors = {o: [] for o in orders}

    for order in orders:
        print(f"  Order {order}...")
        for n_steps in n_steps_grid:
            dt = total_time_t / n_steps
            circ = trotterized_ising_hw_friendly_circuit(
                n_sites=n_sites, J=J, hx=hx, hz=hz,
                dt=dt, reps=n_steps, order=order, dtype=jnp.complex128
            )
            U_approx = circ.to_matrix()
            err = spectral_error(U_exact, U_approx)
            errors[order].append(err)
            # print(f"    n_steps={n_steps:3d}, dt={dt:.4f}, error={err:.2e}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plot_styles = {
        1: {'marker': 'o', 'linestyle': '-', 'label': '1st-order', 'color': 'black'},
        2: {'marker': 's', 'linestyle': '-', 'label': '2nd-order (HW Friendly)', 'color': 'red'},
        4: {'marker': '^', 'linestyle': '-', 'label': '4th-order (Suzuki)', 'color': 'blue'},
    }

    for order, err_list in errors.items():
        style = plot_styles[order]
        plt.loglog(delta_t_grid, err_list, **style)
        
        # Add reference lines
        # Use the last point to anchor the reference line
        prefactor = err_list[-1] / (delta_t_grid[-1]**order)
        ref_y = prefactor * (delta_t_grid**order)
        plt.loglog(delta_t_grid, ref_y, '--', color=style['color'], alpha=0.4, 
                   label=rf'O($\Delta t^{order}$) ref.')

    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'Spectral Error')
    plt.title(f'Ising HW-Friendly Trotter Error Scaling (N={n_sites}, t={total_time_t})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    
    out_dir = pathlib.Path("plots")
    out_dir.mkdir(exist_ok=True)
    plt.savefig(out_dir / "ising_hw_friendly_scaling.png", dpi=300)
    print(f"Plot saved to {out_dir / 'ising_hw_friendly_scaling.png'}")
    
    # Check scaling slope for 2nd order as a sanity check
    log_dt = jnp.log(delta_t_grid)
    log_err = jnp.log(jnp.array(errors[2]))
    slope, _ = np.polyfit(log_dt, log_err, 1)
    print(f"2nd-order scaling slope: {slope:.4f} (expected ~2.0)")
    if slope < 1.8:
         print("WARNING: 2nd-order scaling slope is less than 1.8.")

if __name__ == "__main__":
    test_ising_trotter_scaling()
