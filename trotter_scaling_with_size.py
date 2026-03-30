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

def run_scaling_experiment():
    # 1. Setup Parameters
    # We use a small fixed time and dt to ensure we are in the perturbative regime
    # where Error ~ C * N * dt^k
    Jx, Jy, Jz = 1.0, 0.8, 0.5
    hx, hy, hz = 0.75, 0.0, 0.6
    total_time = 0.1 
    dt = 0.005
    
    # System sizes to sweep
    # Note: Exact diagonalization (2^N) gets slow around N=12
    n_sites_list = [4, 6, 8, 10]
    
    orders = [1, 2]
    
    print(f"Scaling Experiment: T={total_time}, dt={dt}")
    print(f"Comparing Error vs N for N={n_sites_list}")

    results = {
        "Standard": {o: [] for o in orders},
        "HardwareFriendly": {o: [] for o in orders}
    }
    
    for n_sites in n_sites_list:
        # Calculate number of steps for this N (fixed dt)
        reps = int(np.ceil(total_time / dt))
        actual_dt = total_time / reps
        
        print(f"  Running N={n_sites} (reps={reps}, dt={actual_dt:.4f})...")
        
        # 1. Exact Evolution
        model = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                         hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
        H_full = model.build_hamiltonian_matrix()
        U_exact = expm(-1j * total_time * H_full)
        
        for order in orders:
            # 2. Standard Trotter
            circ_std = trotterized_xyz_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=actual_dt, reps=reps, dtype=jnp.complex128
            )
            U_std = circ_std.to_matrix()
            # Spectral Norm Error
            err_std = float(np.linalg.norm(U_exact - U_std, ord=2))
            results["Standard"][order].append(err_std)
            
            # 3. Hardware Friendly Trotter
            circ_hw = trotterized_hardware_friendly_xyz_circuit(
                n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
                order=order, dt=actual_dt, reps=reps, collapse=True, dtype=jnp.complex128
            )
            U_hw = circ_hw.to_matrix()
            err_hw = float(np.linalg.norm(U_exact - U_hw, ord=2))
            results["HardwareFriendly"][order].append(err_hw)

    # 4. Plotting
    plt.figure(figsize=(10, 6))
    
    colors = {1: 'blue', 2: 'green'}
    markers = {"Standard": 'o', "HardwareFriendly": 's'}
    linestyles = {"Standard": '-', "HardwareFriendly": '--'}
    
    for order in orders:
        c = colors[order]
        
        # Plot Standard
        y_std = np.array(results["Standard"][order])
        plt.plot(n_sites_list, y_std, marker=markers["Standard"], linestyle=linestyles["Standard"], color=c, label=f"Standard (Ord {order})")
        
        # Plot HW Friendly
        y_hw = np.array(results["HardwareFriendly"][order])
        plt.plot(n_sites_list, y_hw, marker=markers["HardwareFriendly"], linestyle=linestyles["HardwareFriendly"], color=c, label=f"HW Friendly (Ord {order})")
        
    plt.xlabel('System Size $N$')
    plt.ylabel(r'Spectral Error $\|U_{exact} - U_{approx}\|_2$')
    plt.title(f'Trotter Error Scaling with System Size (T={total_time}, dt={dt})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(__file__), "trotter_size_scaling.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")
    
    # Print slopes
    print("\nApproximate Slopes (Error / N):")
    for order in orders:
        slope_std = (results["Standard"][order][-1] - results["Standard"][order][0]) / (n_sites_list[-1] - n_sites_list[0])
        slope_hw = (results["HardwareFriendly"][order][-1] - results["HardwareFriendly"][order][0]) / (n_sites_list[-1] - n_sites_list[0])
        print(f"Order {order}: Standard ~ {slope_std:.2e} * N, HW-Friendly ~ {slope_hw:.2e} * N")

if __name__ == "__main__":
    run_scaling_experiment()