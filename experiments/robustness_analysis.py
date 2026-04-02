import rqcopt_mpo.jax_config
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.tensor_network.core_ops_helpers import hs_inner_product_from_mpo
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.circuit.noise import apply_parameter_noise
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import rzz_decompose_ising_circuit
from experiments.utils import get_reference_path

# Parameters (must match the optimization runs)
n_sites = 6
J = 0.0
D = 1.0
hx = 0.75
hz = 0.6
t = 2.0
reps_ref = 20
order_ref = 4
reps_ansatz = 3

# Paths
ref_base_dir = REPO_ROOT / "experiments"
target_path = get_reference_path(
    base_dir=ref_base_dir,
    n_sites=n_sites,
    Jx=J, Jy=J, Jz=D,
    hx=hx, hy=0.0, hz=hz,
    t=t, reps=reps_ref, order=order_ref
)

riemannian_path = REPO_ROOT / "experiments/riemannian_adam/data" / f"circuit_riemannian_reps_{reps_ansatz}.json"
hf_path = REPO_ROOT / "experiments/rzz_ising_experiment/data" / f"circuit_hw_friendly_reps_{reps_ansatz}.json"

# Load data
target_mpo = MPO.load_json(target_path)
circ_riemannian_su4 = Circuit.load_json(riemannian_path)
circ_hf = Circuit.load_json(hf_path)

print(f"Loaded target MPO from {target_path}")
print(f"Loaded Riemannian SU(4) circuit from {riemannian_path}")
print(f"Loaded Hardware-Friendly circuit from {hf_path}")

# Synthesize Riemannian into RZZ basis
print("Synthesizing Riemannian SU(4) circuit into RZZ + U3 basis...")
circ_riemannian_rzz = rzz_decompose_ising_circuit(circ_riemannian_su4)
print(f"Synthesized Riemannian circuit has {circ_riemannian_rzz.num_layers} layers")

# Check noiseless losses
def compute_loss(circ, target_mpo):
    circ_mpo = circuit_to_mpo(circ)
    overlap = hs_inner_product_from_mpo(circ_mpo, target_mpo)
    return overlap_to_loss(overlap, n_sites=n_sites, normalize=True)

loss_riemannian_noiseless = compute_loss(circ_riemannian_rzz, target_mpo)
loss_hf_noiseless = compute_loss(circ_hf, target_mpo)

print(f"Noiseless Loss (Riemannian-RZZ): {loss_riemannian_noiseless:.6e}")
print(f"Noiseless Loss (HF):             {loss_hf_noiseless:.6e}")

# Robustness analysis
noise_levels = jnp.logspace(-4, -1, 10)
n_samples = 50 # Increase for smoother curves
results_riemannian = []
results_hf = []

print(f"Starting Monte Carlo simulation with {n_samples} samples per noise level...")

for sigma in noise_levels:
    losses_riemannian = []
    losses_hf = []
    
    for i in range(n_samples):
        # Apply noise to synthesized Riemannian
        noisy_riemannian = apply_parameter_noise(circ_riemannian_rzz, sigma, seed=None)
        losses_riemannian.append(compute_loss(noisy_riemannian, target_mpo))
        
        # Apply noise to HF
        noisy_hf = apply_parameter_noise(circ_hf, sigma, seed=None)
        losses_hf.append(compute_loss(noisy_hf, target_mpo))
    
    avg_r = np.mean(losses_riemannian)
    avg_hf = np.mean(losses_hf)
    results_riemannian.append(avg_r)
    results_hf.append(avg_hf)
    print(f"Sigma: {sigma:.1e} | Riemannian Loss: {avg_r:.6e} | HF Loss: {avg_hf:.6e}")

# Plotting
plt.figure(figsize=(8, 6))
plt.loglog(noise_levels, results_riemannian, 'o-', label='Riemannian (Synthesized to RZZ)')
plt.loglog(noise_levels, results_hf, 's-', label='Hardware-Friendly (Optimized in RZZ)')
plt.axhline(y=loss_riemannian_noiseless, color='blue', linestyle='--', alpha=0.5, label='Riemannian Noiseless')
plt.axhline(y=loss_hf_noiseless, color='orange', linestyle='--', alpha=0.5, label='HF Noiseless')

plt.xlabel('Parameter Noise Magnitude ($\sigma$)')
plt.ylabel('Average Fidelity Loss (HST)')
plt.title(f'Robustness to Parameter Noise ($n={n_sites}$, $t={t}$, reps={reps_ansatz}$)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plot_path = REPO_ROOT / "plots/robustness_analysis.png"
plot_path.parent.mkdir(exist_ok=True)
plt.savefig(plot_path)
print(f"Saved plot to {plot_path}")

# Save data for the article narrative
np.savez(REPO_ROOT / "experiments/data/robustness_results.npz",
         noise_levels=noise_levels,
         results_riemannian=results_riemannian,
         results_hf=results_hf,
         loss_riemannian_noiseless=loss_riemannian_noiseless,
         loss_hf_noiseless=loss_hf_noiseless)
