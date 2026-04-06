"""
Qiskit Robustness Simulation
=============================
Post-optimization noise resilience analysis comparing Riemannian-synthesized
circuits against Hardware-Friendly (HF) ansatz circuits under realistic CPTP
noise channels (depolarizing + amplitude damping).

This script:
  1. Loads pre-optimized circuits (or generates initial Trotter stand-ins).
  2. Synthesizes Riemannian SU(4) blocks into the RZZ+U3 basis.
  3. Converts both circuits to qiskit.QuantumCircuit via circuit_to_qiskit.
  4. Verifies the export matches Circuit.to_matrix() in the noiseless limit.
  5. Sweeps over noise scaling factors using qiskit_aer DensityMatrixSimulator.
  6. Computes state fidelity relative to each circuit's own noiseless output,
     isolating the noise-resilience from the ansatz approximation error.
  7. Saves results and generates publication-quality plots.

Usage:
    source ../../environments/qtools/bin/activate
    python experiments/qiskit_robustness_simulation.py
"""
import rqcopt_mpo.jax_config

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder import (
    rzz_decompose_ising_circuit,
)
from rqcopt_mpo.utils.qiskit_export import circuit_to_qiskit
from experiments.utils import get_reference_path

# ──────────────────────────────────────────────────────────────
# 0. Hamiltonian / experiment parameters  (must match optimization runs)
# ──────────────────────────────────────────────────────────────
N_SITES   = 6
J         = 0.0     # Jx = Jy = 0 (Ising model)
D         = 1.0     # Jz
HX        = 0.75
HZ        = 0.6
T_TOTAL   = 2.0
REPS_REF  = 20
ORDER_REF = 4
REPS_ANSATZ = 3

# Noise sweep parameters
NOISE_LEVELS   = np.logspace(-4, -1, 12)   # depolarizing probability p
AMP_DAMP_RATIO = 0.5                        # p_damp = p * AMP_DAMP_RATIO
TWO_Q_FACTOR   = 10                         # 2Q depol error = p * TWO_Q_FACTOR

# Initial states to evaluate (indices into the computational basis)
INITIAL_STATES = {
    "|000000⟩": 0,
    "|010101⟩": int("010101", 2),
}

# Output paths
PLOT_DIR = REPO_ROOT / "plots"
DATA_DIR = REPO_ROOT / "experiments" / "data"

# ──────────────────────────────────────────────────────────────
# 1. Data loading helpers
# ──────────────────────────────────────────────────────────────

def _load_or_generate_circuits():
    """
    Try to load pre-optimized circuits. If they don't exist, fall back to
    un-optimized initial Trotter circuits so the pipeline can be tested.
    """
    riem_path = (
        REPO_ROOT / "experiments/riemannian_adam/data"
        / f"circuit_riemannian_reps_{REPS_ANSATZ}.json"
    )
    hf_path = (
        REPO_ROOT / "experiments/rzz_ising_experiment/data"
        / f"circuit_hw_friendly_reps_{REPS_ANSATZ}.json"
    )

    fallback = False

    # --- Riemannian circuit ---
    if riem_path.exists():
        circ_riem = Circuit.load_json(riem_path)
        print(f"[OK] Loaded Riemannian circuit from {riem_path}")
    else:
        print(f"[WARN] Riemannian circuit not found at {riem_path}")
        print("       Generating initial Trotter (SU(4)-block) circuit as stand-in...")
        from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
            trotterized_xyz_circuit,
        )
        dt = T_TOTAL / REPS_ANSATZ
        circ_riem = trotterized_xyz_circuit(
            n_sites=N_SITES, Jx=J, Jy=J, Jz=D,
            hx=HX, hz=HZ, dt=dt, reps=REPS_ANSATZ, order=2,
            method="suzuki", dtype=jnp.complex128,
        )
        fallback = True

    # --- HF circuit ---
    if hf_path.exists():
        circ_hf = Circuit.load_json(hf_path)
        print(f"[OK] Loaded HF circuit from {hf_path}")
    else:
        print(f"[WARN] HF circuit not found at {hf_path}")
        print("       Generating initial Trotter (HW-friendly) circuit as stand-in...")
        from rqcopt_mpo.circuit.trotter.trotter_ising_hw_friendly import (
            trotterized_ising_hw_friendly_circuit,
        )
        from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import (
            rzz_decompose_ising_circuit as rzz_decompose_hw,
        )
        dt = T_TOTAL / REPS_ANSATZ
        raw = trotterized_ising_hw_friendly_circuit(
            n_sites=N_SITES, J=D, hx=HX, hz=HZ,
            dt=dt, reps=REPS_ANSATZ, order=2,
            dtype=jnp.complex128,
        )
        circ_hf = rzz_decompose_hw(raw, order=2)
        fallback = True

    if fallback:
        print("\n*** Running with fall-back Trotter circuits (not optimized). ***")
        print("*** Run the optimization experiments first for meaningful results. ***\n")

    return circ_riem, circ_hf, fallback


def _load_target_mpo():
    """Load the reference MPO, or return None if unavailable."""
    ref_base_dir = REPO_ROOT / "experiments"
    target_path = get_reference_path(
        base_dir=ref_base_dir, n_sites=N_SITES,
        Jx=J, Jy=J, Jz=D, hx=HX, hy=0.0, hz=HZ,
        t=T_TOTAL, reps=REPS_REF, order=ORDER_REF,
    )
    if target_path.exists():
        mpo = MPO.load_json(target_path)
        print(f"[OK] Loaded target MPO from {target_path}")
        return mpo
    else:
        print(f"[WARN] Target MPO not found at {target_path} — skipping absolute fidelity.")
        return None


# ──────────────────────────────────────────────────────────────
# 2. Circuit export verification
# ──────────────────────────────────────────────────────────────

def verify_export(circ: Circuit, label: str, atol: float = 1e-10):
    """
    Verify that circuit_to_qiskit produces a unitary that matches
    Circuit.to_matrix() in the noiseless limit.
    """
    U_tn = np.array(circ.to_matrix())
    qc = circuit_to_qiskit(circ)
    U_qiskit = Operator(qc).data

    diff = np.linalg.norm(U_tn - U_qiskit)
    ok = diff < atol

    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}: ||U_tn - U_qiskit|| = {diff:.2e}  (tol={atol:.0e})")
    if not ok:
        # Check up to global phase
        phases = U_qiskit.conj().T @ U_tn
        phase_diag = np.diag(phases)
        phase_std = np.std(np.angle(phase_diag))
        print(f"         phase-angle std = {phase_std:.2e}  "
              f"(if ~0 → global-phase mismatch only)")
    return ok, qc


# ──────────────────────────────────────────────────────────────
# 3. Noise modelling helpers
# ──────────────────────────────────────────────────────────────

def _build_noise_model(p_depol: float, p_damp: float) -> NoiseModel:
    """
    Build a realistic hardware noise model:
      • 1Q gates (rz, ry): depolarizing(p) ⊗ amplitude_damping(p_damp)
      • 2Q gates (rzz):    depolarizing(p*10) ⊗ amp_damp(p_damp)^{⊗2}
    """
    noise_model = NoiseModel()

    # 1-qubit channel
    error_1q = depolarizing_error(p_depol, 1).compose(
        amplitude_damping_error(p_damp)
    )

    # 2-qubit channel
    error_2q = depolarizing_error(
        min(p_depol * TWO_Q_FACTOR, 1.0), 2       # clamp to valid range
    ).compose(
        amplitude_damping_error(p_damp).tensor(amplitude_damping_error(p_damp))
    )

    noise_model.add_all_qubit_quantum_error(error_1q, ["rz", "ry"])
    noise_model.add_all_qubit_quantum_error(error_2q, ["rzz"])
    return noise_model


# ──────────────────────────────────────────────────────────────
# 4. Simulation functions
# ──────────────────────────────────────────────────────────────

# Use density_matrix method  — exact for N=6 (64×64)
_SIMULATOR = AerSimulator(method="density_matrix")


def get_noiseless_statevector(qc: QuantumCircuit) -> Statevector:
    """Run a noiseless density-matrix simulation, return Statevector."""
    return Statevector.from_instruction(qc)


from typing import Optional, Dict, Tuple, Any

def run_noisy_simulation(
    qc: QuantumCircuit,
    noise_model: NoiseModel,
    initial_state: Optional[Statevector] = None,
) -> DensityMatrix:
    """Simulate circuit under noise, return final DensityMatrix."""
    qc_sim = qc.copy()
    qc_sim.save_density_matrix()
    result = _SIMULATOR.run(
        qc_sim,
        noise_model=noise_model,
    ).result()
    return result.data()["density_matrix"]


def compute_fidelity(
    rho_noisy: DensityMatrix,
    psi_clean: Statevector,
) -> float:
    """F = ⟨ψ_clean| ρ_noisy |ψ_clean⟩"""
    return float(state_fidelity(psi_clean, rho_noisy))


# ──────────────────────────────────────────────────────────────
# 5. Main robustness sweep
# ──────────────────────────────────────────────────────────────

def robustness_sweep(
    qc_riem: QuantumCircuit,
    qc_hf: QuantumCircuit,
    noise_levels: np.ndarray,
):
    """
    For each noise level, simulate both circuits under noise and compute
    state fidelity relative to each circuit's own noiseless output.

    Returns
    -------
    results : dict  with keys:
        noise_levels, fid_riem, fid_hf, depth_riem, depth_hf,
        gate_counts_riem, gate_counts_hf
    """
    # Get noiseless references
    psi_riem_clean = get_noiseless_statevector(qc_riem)
    psi_hf_clean   = get_noiseless_statevector(qc_hf)

    fid_riem = np.zeros(len(noise_levels))
    fid_hf   = np.zeros(len(noise_levels))

    print(f"\n{'p_depol':>10s}  {'Fid(Riem)':>10s}  {'Fid(HF)':>10s}")
    print("-" * 36)

    for i, p in enumerate(noise_levels):
        p_damp = p * AMP_DAMP_RATIO
        nm = _build_noise_model(p, p_damp)

        rho_r = run_noisy_simulation(qc_riem, nm)
        rho_h = run_noisy_simulation(qc_hf, nm)

        fid_riem[i] = compute_fidelity(rho_r, psi_riem_clean)
        fid_hf[i]   = compute_fidelity(rho_h, psi_hf_clean)

        print(f"  {p:8.2e}    {fid_riem[i]:.6f}    {fid_hf[i]:.6f}")

    # Circuit metadata
    gate_counts_riem = dict(qc_riem.count_ops())
    gate_counts_hf   = dict(qc_hf.count_ops())

    return {
        "noise_levels": noise_levels,
        "fid_riem": fid_riem,
        "fid_hf": fid_hf,
        "depth_riem": qc_riem.depth(),
        "depth_hf": qc_hf.depth(),
        "gate_counts_riem": gate_counts_riem,
        "gate_counts_hf": gate_counts_hf,
    }


# ──────────────────────────────────────────────────────────────
# 6. Plotting
# ──────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: Path):
    """Publication-quality fidelity-vs-noise plot."""
    nl = results["noise_levels"]
    fr = results["fid_riem"]
    fh = results["fid_hf"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.semilogx(nl, fr, "o-", color="#d62728", linewidth=2, markersize=6,
                label=f'Riemannian (depth={results["depth_riem"]})')
    ax.semilogx(nl, fh, "s-", color="#1f77b4", linewidth=2, markersize=6,
                label=f'Hardware-Friendly (depth={results["depth_hf"]})')

    ax.set_xlabel("Base Depolarizing Probability $p$", fontsize=13)
    ax.set_ylabel("State Fidelity  $F(\\rho_{noisy}, |\\psi_{clean}\\rangle)$",
                   fontsize=13)
    ax.set_title(
        f"Robustness to Decoherence  ($N={N_SITES}$, reps={REPS_ANSATZ})",
        fontsize=14,
    )
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)

    # Annotate gate counts
    gc_r = results["gate_counts_riem"]
    gc_h = results["gate_counts_hf"]
    info_r = ", ".join(f"{k}={v}" for k, v in sorted(gc_r.items()))
    info_h = ", ".join(f"{k}={v}" for k, v in sorted(gc_h.items()))
    ax.text(
        0.98, 0.02,
        f"Riemannian gates: {info_r}\nHF gates: {info_h}",
        transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


# ──────────────────────────────────────────────────────────────
# 7. Entry point
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Qiskit Robustness Analysis  —  Post-Optimization Noise")
    print("=" * 60)

    # ---- Load circuits ------------------------------------------------
    circ_riem_raw, circ_hf, is_fallback = _load_or_generate_circuits()
    target_mpo = _load_target_mpo()

    # ---- Synthesize Riemannian SU(4) blocks into RZZ + U3 basis -------
    print("\nSynthesizing Riemannian circuit into RZZ + U3 basis...")
    circ_riem_rzz = rzz_decompose_ising_circuit(circ_riem_raw)
    print(f"  Riemannian-RZZ layers = {circ_riem_rzz.num_layers}")
    print(f"  HF layers             = {circ_hf.num_layers}")

    # ---- Convert to Qiskit QuantumCircuits ----------------------------
    print("\nConverting to Qiskit QuantumCircuits...")
    qc_riem = circuit_to_qiskit(circ_riem_rzz)
    qc_hf   = circuit_to_qiskit(circ_hf)
    print(f"  qc_riem depth={qc_riem.depth()}, ops={dict(qc_riem.count_ops())}")
    print(f"  qc_hf   depth={qc_hf.depth()},   ops={dict(qc_hf.count_ops())}")

    # ---- Verification -------------------------------------------------
    print("\n--- Export verification (noiseless) ---")
    ok_r, _ = verify_export(circ_riem_rzz, "Riemannian-RZZ")
    ok_h, _ = verify_export(circ_hf,       "Hardware-Friendly")
    if not (ok_r and ok_h):
        print("\n⚠  Export verification failed — results may be unreliable.\n")

    # ---- Noise sweep --------------------------------------------------
    print("\n--- Noise sweep ---")
    results = robustness_sweep(qc_riem, qc_hf, NOISE_LEVELS)

    # ---- Save numerical results ---------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = DATA_DIR / "qiskit_robustness_results.npz"
    np.savez(
        npz_path,
        noise_levels=results["noise_levels"],
        fid_riem=results["fid_riem"],
        fid_hf=results["fid_hf"],
        depth_riem=results["depth_riem"],
        depth_hf=results["depth_hf"],
        n_sites=N_SITES,
        reps_ansatz=REPS_ANSATZ,
        is_fallback=is_fallback,
    )
    print(f"Results saved to {npz_path}")

    # ---- Plot ---------------------------------------------------------
    plot_path = PLOT_DIR / "qiskit_robustness.png"
    plot_results(results, plot_path)

    # ---- Summary table ------------------------------------------------
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  N_sites          = {N_SITES}")
    print(f"  Reps (ansatz)    = {REPS_ANSATZ}")
    print(f"  Riemannian depth = {results['depth_riem']}")
    print(f"  HF depth         = {results['depth_hf']}")
    print(f"  Noise levels     = {len(NOISE_LEVELS)} points in "
          f"[{NOISE_LEVELS[0]:.0e}, {NOISE_LEVELS[-1]:.0e}]")
    print(f"  Using fallback?  = {is_fallback}")

    # Fidelity at strongest noise
    worst_idx = -1
    print(f"\n  Fidelity at p={NOISE_LEVELS[worst_idx]:.1e}:")
    print(f"    Riemannian : {results['fid_riem'][worst_idx]:.6f}")
    print(f"    HF         : {results['fid_hf'][worst_idx]:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
