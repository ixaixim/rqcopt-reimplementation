# Robustness Analysis: Post-Optimization Qiskit Simulation

## Background & Motivation
The current tensor network optimizer effectively finds high-fidelity unitary approximations of time-evolution operators. However, we need to prove the "Strategic Pivot": the Hardware-Friendly (HF) ansatz restricts the search space to native topologies, naturally avoiding noise-sensitive parameter regimes. Conversely, the Riemannian approach optimizes monolithic $SU(4)$ blocks that, when synthesized, often result in deep circuits or highly sensitive rotation angles.

To make a compelling argument for the article, we must demonstrate that the HF approach achieves significantly higher physical fidelity under realistic noise (amplitude damping, depolarizing) than the blindly compiled Riemannian code.

## Scope & Impact
Injecting true completely positive trace-preserving (CPTP) noise (like amplitude damping) directly into the tensor network optimization loop would require rewriting the engine to support superoperators, squaring the physical dimensions ($d=2 \to d=4$). This is computationally prohibitive.

Instead, we will perform a **Post-Optimization Simulation**. We will export the pre-optimized unitary circuits into Qiskit and simulate them under realistic CPTP noise channels using Qiskit Aer. This proves the hypothesis without altering the core unitary Tensor Network optimization engine.

## Proposed Solution

### 1. Circuit Translation (`circuit_to_qiskit`)
Create a utility to map our `Circuit` dataclass (and specifically the synthesized native gates `Ising_hw_1q`, `Ising_hw_1rzz`, `u3`, `rzz`) into a `qiskit.QuantumCircuit`. Qiskit is already a project dependency and heavily used for decompositions.

### 2. Qiskit Aer Noise Model
Define a realistic hardware noise model using `qiskit_aer.noise`:
*   **Depolarizing Error:** Applied to single-qubit (`U3`) and two-qubit (`RZZ`) gates.
*   **Amplitude Damping / Thermal Relaxation:** Derived from realistic $T_1$ and $T_2$ times.

### 3. State Fidelity Simulation
Since Hilbert-Schmidt distance for mixed states (Quantum Process Tomography) is computationally expensive, we will evaluate the circuits on a set of initial states (e.g., the Neel state $|010101\rangle$ or $|000000\rangle$).
*   Calculate the exact ideal final state vector $|\psi_{\text{exact}}\rangle$ using the target MPO or exact Hamiltonian evolution.
*   Simulate the density matrix $\rho_{\text{noisy}}$ using `qiskit_aer` with the noise model.
*   Compute the state fidelity: $F = \langle \psi_{\text{exact}} | \rho_{\text{noisy}} | \psi_{\text{exact}} \rangle$.

## Implementation Steps

1.  **Create Exporter:** Add `circuit_to_qiskit(circ: Circuit) -> QuantumCircuit` in `rqcopt_mpo/utils/qiskit_export.py` (or similar utility file). It will iterate over layers and map parameters to Qiskit's `qc.u()` and `qc.rzz()`.
2.  **Simulation Script:** Create `experiments/qiskit_robustness_simulation.py` (or extend `robustness_analysis.py`).
    *   Load the optimized `circuit_riemannian_reps_X.json` and `circuit_hw_friendly_reps_X.json`.
    *   Synthesize the Riemannian circuit into the RZZ basis (already done via `rzz_decompose_ising_circuit`).
    *   Convert both to `QuantumCircuit`.
3.  **Noise Sweep:** Iterate over scaling factors of a baseline noise model (e.g., increasing depolarizing probability).
4.  **Evaluation:** Use `qiskit_aer`'s `DensityMatrixSimulator` (fast for $N=6$ qubits) to get the final density matrix and calculate state fidelity against the exact target state.
5.  **Plotting:** Generate a plot comparing the state fidelity of Riemannian-Synthesized vs Hardware-Friendly under increasing noise levels.

## Alternatives Considered
*   **Full Superoperator TN Rewrite:** Rewriting the MPO and contraction engine to support superoperators (Choi representation). *Rejected* due to extreme complexity and performance hits.
*   **Optimization under Coherent Noise:** Injecting stochastic parameter noise during optimization via `jax.vmap`. *Considered secondary* because it only models control noise, not realistic decoherence (amplitude damping), which is the core of the NISQ argument.

## Verification
*   Verify that `circuit_to_qiskit` produces a unitary exactly matching the `Circuit.to_matrix()` output in the noiseless limit.
*   Verify that the fidelity of the HF approach degrades slower than the Riemannian approach under the same noise model.