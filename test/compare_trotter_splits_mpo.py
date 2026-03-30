import sys
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Ensure rqcopt_mpo is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import trotterized_ising_hw_friendly_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.tensor_network.core_ops_helpers import hs_inner_product_from_mpo

def spectral_error_mpo(mpo_ref, mpo_approx, n_sites):
    """
    Approximate spectral error using MPO inner products.
    ||U_ref - U_approx||_2 approx sqrt(||U_ref - U_approx||_F^2 / d^N)
    ||A - B||_F^2 = <A, A> + <B, B> - 2Re<A, B>
    Since both are nearly unitary, <A,A> approx <B,B> approx 2^N.
    """
    hs_ref = jnp.real(hs_inner_product_from_mpo(mpo_ref, mpo_ref))
    hs_approx = jnp.real(hs_inner_product_from_mpo(mpo_approx, mpo_approx))
    overlap = hs_inner_product_from_mpo(mpo_ref, mpo_approx)
    
    frob_sq = hs_ref + hs_approx - 2.0 * jnp.real(overlap)
    # Spectral norm \approx \sqrt{Frob^2 / dim}
    return float(jnp.sqrt(jnp.maximum(frob_sq, 0.0) / (2**n_sites)))

def run_comparison():
    # 1. Setup Parameters
    n_sites = 10
    Jx, Jy, Jz = 0.0, 0.0, 1.0 # Ising model
    hx, hy, hz = 0.75, 0.0, 0.6
    total_time = 2.0
    max_bondim = 128
    
    # Reference parameters
    ref_order = 4
    ref_reps = 20
    
    print(f"Comparing Trotter Methods via MPO for Ising Model, N={n_sites}, T={total_time}")
    print(f"Couplings: J={Jz}, h=({hx},{hz})")
    print(f"Reference: {ref_reps} reps of {ref_order}th order Standard Trotter")

    # 2. Reference MPO
    print("Computing reference MPO...")
    # Standard split: fields divided between even and odd layers.
    # trotterized_xyz_circuit uses this standard split.
    circ_ref = trotterized_xyz_circuit(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
        order=ref_order, dt=total_time/ref_reps, reps=ref_reps, dtype=jnp.complex128
    )
    mpo_ref = circuit_to_mpo(circ_ref, max_bondim=max_bondim)

    # 3. Sweep parameters
    n_steps_list = np.unique(np.logspace(0.6, 2.0, 10).astype(int))
    dt_list = total_time / n_steps_list
    
    results = {
        "Standard": [],
        "XYZ-HW": [],
        "Ising-HW": []
    }

    print(f"Sweeping over {len(n_steps_list)} time steps...")
    for n_steps in n_steps_list:
        dt = total_time / n_steps
        print(f"  n_steps={n_steps:3d}, dt={dt:.4f}")
        
        # --- Standard Method (2nd order) ---
        circ_std = trotterized_xyz_circuit(
            n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
            order=2, dt=dt, reps=n_steps, dtype=jnp.complex128
        )
        mpo_std = circuit_to_mpo(circ_std, max_bondim=max_bondim)
        err_std = spectral_error_mpo(mpo_ref, mpo_std, n_sites)
        results["Standard"].append(err_std)
        
        # --- XYZ Hardware Friendly Method (2nd order) ---
        circ_xyz_hw = trotterized_hardware_friendly_xyz_circuit(
            n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz,
            order=2, dt=dt, reps=n_steps, collapse=True, dtype=jnp.complex128
        )
        mpo_xyz_hw = circuit_to_mpo(circ_xyz_hw, max_bondim=max_bondim)
        err_xyz_hw = spectral_error_mpo(mpo_ref, mpo_xyz_hw, n_sites)
        results["XYZ-HW"].append(err_xyz_hw)

        # --- Ising Hardware Friendly Method (2nd order) ---
        circ_ising_hw = trotterized_ising_hw_friendly_circuit(
            n_sites=n_sites, J=Jz, hx=hx, hz=hz,
            dt=dt, reps=n_steps, order=2, dtype=jnp.complex128
        )
        mpo_ising_hw = circuit_to_mpo(circ_ising_hw, max_bondim=max_bondim)
        err_ising_hw = spectral_error_mpo(mpo_ref, mpo_ising_hw, n_sites)
        results["Ising-HW"].append(err_ising_hw)

    # 4. Analysis & Plotting
    plt.figure(figsize=(10, 7))
    
    plt.loglog(dt_list, results["Standard"], 'o-', color='blue', label="Standard (Ord 2)")
    plt.loglog(dt_list, results["XYZ-HW"], 's--', color='red', label="XYZ-HW (Ord 2)")
    plt.loglog(dt_list, results["Ising-HW"], '^-.', color='green', label="Ising-HW (Ord 2)")
    
    # Reference slope O(dt^2)
    ref_y2 = results["Standard"][-1] * (dt_list / dt_list[-1])**2
    plt.loglog(dt_list, ref_y2, ':', color='gray', alpha=0.5, label='O(dt^2) ref')

    plt.xlabel(r'Time step $\Delta t$')
    plt.ylabel(r'Estimated Spectral Error $\|U_{ref} - U_{approx}\|_2$')
    plt.title(f'Trotter MPO Comparison (Ising Model, N={n_sites}, t={total_time})')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, "compare_trotter_methods_mpo_ising.png")
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    run_comparison()
