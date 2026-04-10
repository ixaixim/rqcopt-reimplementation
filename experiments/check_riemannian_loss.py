import rqcopt_mpo.jax_config
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.tensor_network.core_ops_helpers import hs_inner_product_from_mpo
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import rzz_decompose_ising_circuit
from experiments.utils import get_reference_path, load_experiment_json

def compute_loss(circ, target_mpo, n_sites):
    circ_mpo = circuit_to_mpo(circ)
    overlap = hs_inner_product_from_mpo(circ_mpo, target_mpo)
    return overlap_to_loss(overlap, n_sites=n_sites, normalize=True)

# Parameters
n_sites = 6
J = 0.0
D = 1.0
hx = 0.75
hz = 0.6
t = 2.0
reps_ref = 20
order_ref = 4

# Paths
ref_base_dir = REPO_ROOT / "experiments"
target_path = get_reference_path(
    base_dir=ref_base_dir,
    n_sites=n_sites,
    Jx=J, Jy=J, Jz=D,
    hx=hx, hy=0.0, hz=hz,
    t=t, reps=reps_ref, order=order_ref
)

riemannian_path = REPO_ROOT / "experiments/riemannian_adam/data/run_riemannian-opt_20260410_154421.json"

# Load data
target_mpo = MPO.load_json(target_path)
exp_data = load_experiment_json(riemannian_path)
circ_su4 = exp_data["circuit"]

print(f"Reported final_loss in JSON: {exp_data['final_loss']}")

# Loss of original SU(4) circuit
loss_su4 = compute_loss(circ_su4, target_mpo, n_sites)
print(f"Calculated Loss (Original SU4): {loss_su4:.6e}")

# Synthesize and check again
print("Synthesizing to RZZ...")
circ_rzz = rzz_decompose_ising_circuit(circ_su4)
loss_rzz = compute_loss(circ_rzz, target_mpo, n_sites)
print(f"Calculated Loss (Synthesized RZZ): {loss_rzz:.6e}")
