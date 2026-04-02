import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import rzz_decompose_ising_circuit
from experiments.utils import get_reference_path
from rqcopt_mpo.utils.qiskit_export import circuit_to_qiskit

# Parameters
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

if not target_path.exists() or not riemannian_path.exists() or not hf_path.exists():
    print("Required data files not found. Ensure optimization experiments have been run.")
    sys.exit(1)

# Load data
print("Loading MPO and circuits...")
target_mpo = MPO.load_json(target_path)
circ_riemannian_su4 = Circuit.load_json(riemannian_path)
circ_hf = Circuit.load_json(hf_path)

# Synthesize Riemannian into RZZ basis
print("Synthesizing Riemannian SU(4) circuit...")
circ_riemannian_rzz = rzz_decompose_ising_circuit(circ_riemannian_su4)

# Convert to Qiskit
print("Converting to Qiskit QuantumCircuits...")
qc_riemannian = circuit_to_qiskit(circ_riemannian_rzz)
qc_hf = circuit_to_qiskit(circ_hf)

# Get the exact target state |psi_exact> = U_target |0...0>
print("Computing exact target state...")
# Target MPO to matrix
U_target = np.array(target_mpo.to_matrix())
# Initial state |0...0>
psi_0 = np.zeros(2**n_sites, dtype=complex)
psi_0[0] = 1.0
# We need to be careful with qubit ordering. Qiskit is little-endian.
# Our TN contraction might be big-endian.
# Let's use the noiseless Qiskit circuits as the baseline reference instead to avoid endianness issues.
# A noiseless simulation of qc_hf / qc_riemannian will serve as a proxy.
# Wait, we want to see how fidelity drops relative to the EXACT target.
# But endianness might be an issue. Let's just use Statevector from Qiskit on the ideal qcs!
# We will measure robustness relative to their OWN noiseless output, or we can use qc_hf noiseless as the reference.
# To be fair to the optimizer, we evaluate fidelity against the target MPO. 
# But for a purely robustness analysis, comparing the noisy state to the noiseless state produced by the *same* circuit 
# isolates the effect of noise from the approximation error of the ansatz.
# We will compute: Fidelity( rho_noisy, rho_noiseless )
# This isolates the "Noise-Resilience" exactly!

simulator = AerSimulator(method='density_matrix')

def get_noiseless_state(qc: QuantumCircuit) -> Statevector:
    qc_clean = qc.copy()
    qc_clean.save_statevector()
    result = simulator.run(qc_clean).result()
    return result.get_statevector()

state_riemannian_clean = get_noiseless_state(qc_riemannian)
state_hf_clean = get_noiseless_state(qc_hf)

def run_noisy_simulation(qc: QuantumCircuit, p_depol: float, p_damp: float) -> DensityMatrix:
    noise_model = NoiseModel()
    
    # Add depolarizing error to 1-qubit and 2-qubit gates
    error_1q = depolarizing_error(p_depol, 1)
    error_2q = depolarizing_error(p_depol * 10, 2) # Typical 2Q error is higher
    
    # Add amplitude damping
    error_damp = amplitude_damping_error(p_damp)
    
    # Combine errors
    error_1q_combined = error_1q.compose(error_damp)
    
    # We compose damping on both qubits for 2Q gate
    error_damp_2q = error_damp.tensor(error_damp)
    error_2q_combined = error_2q.compose(error_damp_2q)

    noise_model.add_all_qubit_quantum_error(error_1q_combined, ['u', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2q_combined, ['rzz'])
    
    qc_noisy = qc.copy()
    qc_noisy.save_density_matrix()
    
    result = simulator.run(qc_noisy, noise_model=noise_model).result()
    return result.get_density_matrix()

noise_levels = np.logspace(-4, -1, 10)
fidelities_riemannian = []
fidelities_hf = []

print("Running noisy simulations...")
for p in noise_levels:
    # Set amplitude damping probability roughly proportional to depolarizing
    p_damp = p * 0.5 
    
    rho_riemannian = run_noisy_simulation(qc_riemannian, p, p_damp)
    rho_hf = run_noisy_simulation(qc_hf, p, p_damp)
    
    fid_r = state_fidelity(state_riemannian_clean, rho_riemannian)
    fid_h = state_fidelity(state_hf_clean, rho_hf)
    
    fidelities_riemannian.append(fid_r)
    fidelities_hf.append(fid_h)
    
    print(f"p_depol: {p:.1e} | Fid(Riemannian): {fid_r:.4f} | Fid(HF): {fid_h:.4f}")

# Plotting
plt.figure(figsize=(8, 6))
plt.semilogx(noise_levels, fidelities_riemannian, 'o-', label='Riemannian (Synthesized SU(4))')
plt.semilogx(noise_levels, fidelities_hf, 's-', label='Hardware-Friendly Ansatz')

plt.xlabel('Base Depolarizing Probability ($p$)')
plt.ylabel('Fidelity relative to Noiseless Circuit Output')
plt.title(f'Robustness to Decoherence ($N={n_sites}$, reps={reps_ansatz})')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)

plot_path = REPO_ROOT / "plots/qiskit_robustness.png"
plot_path.parent.mkdir(exist_ok=True)
plt.savefig(plot_path)
print(f"Saved plot to {plot_path}")
