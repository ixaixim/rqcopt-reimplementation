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

def check_gradient_sensitivity():
    print("--- Setting up Gradient Sensitivity Check ---")
    
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
    # We use 0.0 as the baseline (highest precision)
    cutoffs = [0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4]
    max_bondim_env = 128 # Keep this fixed and high to isolate SVD cutoff effect
    
    results = {}
    baseline_grads = None
    baseline_grads_list = None
    baseline_norm = 1.0 # default to avoid unbound variable issues

    print("\n--- Running Sweep ---")
    print(f"{'SVD Cutoff':<15} | {'Trace (Loss Proxy)':<20} | {'Rel Grad Diff':<15} | {'Abs Grad Diff':<15}")
    print("-" * 60)
    
    for cutoff in cutoffs:
        # Compute gradient
        # We use 'bottom-up' sweep; results should be consistent provided we stick to one direction
        loss_val, grads, _ = cost_and_euclidean_grad(
            circuit,
            target_mpo,
            max_bondim_env=max_bondim_env,
            svd_cutoff=cutoff,
            vertical_sweep='top-down'
        )
        
        # Flatten list of gradients into a single vector for comparison
        flat_grad = jnp.concatenate([g.ravel() for g in grads])
        
        if cutoff == 0.0:
            baseline_grads = flat_grad
            baseline_grads_list = grads
            baseline_norm = jnp.linalg.norm(baseline_grads) + 1e-15 # Avoid div/0
            diff = 0.0
            rel_diff = 0.0
        else:
            # Absolute difference (Euclidean/Frobenius norm)
            diff = jnp.linalg.norm(flat_grad - baseline_grads)
            # Relative difference (Unitless, easier to interpret)
            rel_diff = diff / baseline_norm
            
            # Compute per-gate error for visualization
            gate_errors = [float(jnp.linalg.norm(g - bg)) for g, bg in zip(grads, baseline_grads_list)]
            
            # Plot heatmap
            fig, ax = plot_circuit_heatmap(
                circuit, 
                gate_values=gate_errors, 
                title=f"Gradient Error per Gate (Cutoff={cutoff:.1e})",
                log_scale=True
            )
            heatmap_path = Path(__file__).parent / f"gradient_error_heatmap_cutoff_{cutoff:.1e}.pdf"
            fig.savefig(heatmap_path)
            plt.close(fig)
            
        results[cutoff] = (rel_diff, diff)
        print(f"{cutoff:<15.1e} | {loss_val:<20.6f} | {rel_diff:<15.6e} | {diff:<15.6e}")

    # 5. Plotting
    cutoffs_plot = [c for c in cutoffs if c > 0]
    rel_diffs_plot = [results[c][0] for c in cutoffs_plot]
    abs_diffs_plot = [results[c][1] for c in cutoffs_plot]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(cutoffs_plot, rel_diffs_plot, 'o-', label=f'Relative Error (BD={max_bondim_env})')
    plt.loglog(cutoffs_plot, abs_diffs_plot, 's--', label=f'Absolute Error (BD={max_bondim_env})')
    
    plt.xlabel('SVD Cutoff')
    plt.ylabel('Gradient Error (vs cutoff=0.0)')
    plt.title(f'Gradient Sensitivity to SVD Cutoff\n(N={n_sites}, t={t})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    out_path = Path(__file__).parent / "gradient_sensitivity_check.pdf"
    plt.savefig(out_path)
    print(f"\nPlot saved to: {out_path}")
    print("Check the plot: A flat region at low cutoffs indicates stability.")
    print("A sharp rise indicates where the approximation breaks down.")

if __name__ == "__main__":
    check_gradient_sensitivity()