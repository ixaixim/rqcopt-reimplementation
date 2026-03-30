import rqcopt_mpo.jax_config
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import necessary builders and gradient function
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import rzz_decompose_ising_circuit
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.utils_graphics.circuit_plotter import plot_circuit_heatmap

def check_gradient_bond_dim_sensitivity():
    print("--- Setting up Gradient Bond Dimension Sensitivity Check ---")
    
    # 1. Setup System Parameters (Matching rzz_ising_experiment_hw_friendly.py)
    n_sites = 10
    J = 0.
    D = 1.
    hx, hz = 0.75, 0.6
    t = 2.
    dtype = jnp.complex128
    
    # 2. Build Target MPO
    target_reps = 3
    target_order = 4
    target_dt = t / target_reps
    
    print("Building Target MPO...")
    target_circ = trotterized_hardware_friendly_xyz_circuit(    
        n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz,
        order=target_order, dt=target_dt, reps=target_reps, collapse=True,
        dtype=dtype
    )
    target_mpo = circuit_to_mpo(target_circ)
    
    # 3. Build Ansatz Circuit
    ansatz_reps = 3
    ansatz_order = 2
    ansatz_dt = t / ansatz_reps
    
    print("Building Ansatz Circuit...")
    initial_circuit = trotterized_hardware_friendly_xyz_circuit(    
        n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz, 
        order=ansatz_order, dt=ansatz_dt, reps=ansatz_reps, collapse=True,
        dtype=dtype
    )
    
    # Decompose to match the optimizer's view
    circuit = rzz_decompose_ising_circuit(initial_circuit, ansatz_order)
    print(f"Circuit has {circuit.num_layers} layers.")

    # 4. Define Sweep Parameters
    # We use a high bond dimension as the 'Ground Truth' baseline
    baseline_bond_dim = 256
    bond_dims = [2, 4, 8, 16, 32, 64, 128, baseline_bond_dim]
    
    # Fix SVD cutoff to be very small to isolate bond dimension truncation effects
    fixed_svd_cutoff = 0.0
    
    results = {}
    
    print(f"\nComputing Baseline Gradient (Bond Dim={baseline_bond_dim}, Cutoff={fixed_svd_cutoff})...")
    _, baseline_grads_list, _ = cost_and_euclidean_grad(
        circuit,
        target_mpo,
        max_bondim_env=baseline_bond_dim,
        svd_cutoff=fixed_svd_cutoff,
        vertical_sweep='top-down'
    )
    # Flatten for easy vector comparison
    baseline_grads = jnp.concatenate([g.ravel() for g in baseline_grads_list])
    baseline_norm = jnp.linalg.norm(baseline_grads) + 1e-15

    print("\n--- Running Sweep ---")
    print(f"{'Max Bond Dim':<15} | {'Trace (Loss Proxy)':<20} | {'Rel Grad Diff':<15} | {'Abs Grad Diff':<15}")
    print("-" * 60)
    
    for bd in bond_dims:
        # Compute gradient for current bond dimension
        loss_val, grads, _ = cost_and_euclidean_grad(
            circuit,
            target_mpo,
            max_bondim_env=bd,
            svd_cutoff=fixed_svd_cutoff,
            vertical_sweep='bottom-up'
        )
        
        flat_grad = jnp.concatenate([g.ravel() for g in grads])
        
        # Compare to baseline
        diff = jnp.linalg.norm(flat_grad - baseline_grads)
        rel_diff = diff / baseline_norm
        
        results[bd] = (rel_diff, diff)
        print(f"{bd:<15} | {loss_val:<20.6f} | {rel_diff:<15.6e} | {diff:<15.6e}")

        if bd != baseline_bond_dim:
            # Compute per-gate error for visualization
            gate_errors = [float(jnp.linalg.norm(g - bg)) for g, bg in zip(grads, baseline_grads_list)]
            
            # Plot heatmap
            fig, ax = plot_circuit_heatmap(
                circuit, 
                gate_values=gate_errors, 
                title=f"Gradient Error per Gate (Bond Dim={bd})",
                log_scale=True
            )
            heatmap_path = Path(__file__).parent / f"gradient_error_heatmap_bd_{bd}.pdf"
            fig.savefig(heatmap_path)
            plt.close(fig)

    # 5. Plotting
    # Exclude the baseline itself from the plot (error is 0)
    bds_plot = [b for b in bond_dims if b < baseline_bond_dim]
    rel_diffs_plot = [results[c][0] for c in bds_plot]
    abs_diffs_plot = [results[c][1] for c in bds_plot]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(bds_plot, rel_diffs_plot, 'o-', label=f'Relative Error (vs BD={baseline_bond_dim})')
    plt.loglog(bds_plot, abs_diffs_plot, 's--', label=f'Absolute Error (vs BD={baseline_bond_dim})')
    
    plt.xlabel('Max Bond Dimension (Environment)')
    plt.ylabel('Gradient Error')
    plt.title(f'Gradient Sensitivity to Bond Dimension\n(N={n_sites}, t={t}, cutoff={fixed_svd_cutoff})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    out_path = Path(__file__).parent / "gradient_bond_dim_sensitivity.pdf"
    plt.savefig(out_path)
    print(f"\nPlot saved to: {out_path}")

if __name__ == "__main__":
    check_gradient_bond_dim_sensitivity()