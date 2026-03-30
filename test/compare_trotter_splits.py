import sys
import os
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
import matplotlib.pyplot as plt

# Ensure rqcopt_mpo is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import trotterized_ising_hw_friendly_circuit

def run_comparison():
    # 1. Setup Parameters
    n_sites = 6
    Jx, Jy, Jz = 0.0, 0.0, 1.0 # Ising model
    hx, hy, hz = 0.75, 0.0, 0.6
    total_time = 2.0
    
    # Orders to compare
    orders = [2] 
    
    print(f"Comparing Trotter Methods for Ising Model, N={n_sites}, T={total_time}")
    print(f"Couplings: J={Jz}, h=({hx},{hz})")

    # 2. Exact Evolution
    print("Computing exact evolution...")
    model = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                     hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
    H_full = model.build_hamiltonian_matrix()
    U_exact = expm(-1j * total_time * H_full)

    # 3. Sweep parameters
    n_steps_list = np.unique(np.logspace(0.6, 2.0, 15).astype(int))
    dt_list = total_time / n_steps_list
    
    results = {
        "Standard": {o: [] for o in orders},
        "XYZ-HW": {o: [] for o in orders},
        "Ising-HW": {o: [] for o in orders}
    }

    print(f"Sweeping over {len(n_steps_list)} time steps...")
    for n_steps in n_steps_list:
        dt = total_time / n_steps
        
        for order in orders:
            # --- Standard Method ---
            circ_std = trotterized_xyz_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=dt, reps=n_steps, dtype=jnp.complex128
            )
            U_std = circ_std.to_matrix()
            err_std = float(np.linalg.norm(U_exact - U_std, ord=2))
            results["Standard"][order].append(err_std)
            
            # --- XYZ Hardware Friendly Method ---
            circ_xyz_hw = trotterized_hardware_friendly_xyz_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=dt, reps=n_steps, collapse=True, dtype=jnp.complex128
            )
            U_xyz_hw = circ_xyz_hw.to_matrix()
            err_xyz_hw = float(np.linalg.norm(U_exact - U_xyz_hw, ord=2))
            results["XYZ-HW"][order].append(err_xyz_hw)

            # --- Ising Hardware Friendly Method ---
            # New specialized method
            circ_ising_hw = trotterized_ising_hw_friendly_circuit(
                n_sites=n_sites, J=Jz, hx=hx, hz=hz,
                dt=dt, reps=n_steps, dtype=jnp.complex128
            )
            U_ising_hw = circ_ising_hw.to_matrix()
            err_ising_hw = float(np.linalg.norm(U_exact - U_ising_hw, ord=2))
            results["Ising-HW"][order].append(err_ising_hw)

    # 4. Analysis & Plotting
    plt.figure(figsize=(10, 7))
    
    colors = {2: 'green'}
    
    for order in orders:
        c = colors.get(order, 'black')
        plt.loglog(dt_list, results["Standard"][order], 'o-', color='blue', label=f"Standard (Ord {order})")
        plt.loglog(dt_list, results["XYZ-HW"][order], 's--', color='red', label=f"XYZ-HW (Ord {order})")
        plt.loglog(dt_list, results["Ising-HW"][order], '^-.', color='green', label=f"Ising-HW (Ord {order})")
        
    plt.xlabel(r'Time step $\Delta t$')
    plt.ylabel(r'Spectral Error $\|U_{exact} - U_{approx}\|_2$')
    plt.title(f'Trotter Method Comparison (Ising Model, N={n_sites}, t={total_time})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), "compare_trotter_methods_ising.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    run_comparison()