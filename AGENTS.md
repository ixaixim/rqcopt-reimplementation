# General Purpose 
To run compression of quantum circuits using tensor network methods. 
Gradient of loss function with respect to a gate is computed through "punching a hole" in the tensor network (contracting upper, lower, right, left environments).
If internally the gate has parameters, their gradients are computed by chain rule. 
Folders:
experiments/ holds experimnents with different optimizers and circuit decompositions.
circuit/: Brickwall circuit dataclasses, builders, and decomposition utilities (Trotter, CNOT, RZZ Ising, Weyl).
hamiltonian/: Local spin-model Hamiltonians and operator definitions (Heisenberg, XYZ, generic operators).
jax_config.py: JAX configuration helpers (dtype, jit options, global settings).
mpo/: MPO dataclasses and builders for representing time-evolution and other operators as tensor networks.
optimization/: Optimizers, gradient utilities, and experiment-specific optimizers (Riemannian Adam, Weyl, CNOT, RZZ Ising).
tensor_network/: Core tensor-network operations used for contractions and MPO/circuit manipulations.
utils/: General utilities (batched operations, pytrees, rotation helpers, small shared helpers).
# Environment Configuration
When executing Python scripts, ensure the temporary directory is defined and use the project virtual environment. 
ALWAYS run the command using this exact prefix (only if running from VS Code Extension, NOT from Codex CLI):
If you want to run python:
`export TMPDIR=/tmp && /home/neelmiscia/data/environments/rqcopt/bin/python3`


If you want to run a pytest: 
`export TMPDIR=/tmp /home/neelmiscia/data/environments/rqcopt/bin/pytest`
if any of those two commands returns error because of the tmp folder, just run the command without exporting TMPDIR.

**CLI Agents – Best Practice & Code Hygiene:**  
When a question implies a structural change, weigh it against clean-code principles and real-world maintainability. Politeness is fine, but politeness ≠ automatic approval. If the idea adds clutter, coupling, or cleverness at the expense of clarity, say so—and propose a leaner alternative.

# Brickwall circuit and general info
often methods that handle quantum circuits are hardwired for brickwall circuits, or circuits with a specific structure. 
If so, your answer should be tailored to that structure, and should not consider 'if' statements (e.g. if all gates expect a 'params' attribute inside a routine, do not consider an 'if is not gate.params' .)

The code implements different ways of compressing a time evolution quantum circuit operator to match another longer quantum circuit operator. The target operator is expressed as an MPO (Matrix Product Operator). 

# Core dataclasses:
- Circuits are brickwall layers of `GateLayer` objects; each `GateLayer` has `layer_index`, an `is_odd` parity flag that encodes even/odd bonds, and a list of `Gate`s.
- `Circuit.to_matrix()` contracts layers from last to first, building each layer as a kron product over sites (using gate matrices directly, no extra dtype/shape checks beyond the dataclass).
- `Gate` matrices are stored as dense 2x2 or 4x4 arrays; `Gate.tensor` reshapes two-qubit matrices to (2,2,2,2) and is the preferred view for MPO-style contractions.

# MPO conventions:
- `MPO` tensors are 4-legged arrays with shape `(left_bond, phys_out, phys_in, right_bond)`; `MPO.to_matrix()` and canonicalization routines assume this ordering.
- Canonicalisation works via QR sweeps: `left_canonicalize` pushes R to the right, `right_canonicalize` pushes L to the left; `normalize()` only adjusts the edge tensor and `norm` assuming one of these forms.

# Trotter circuits:
- `trotterized_heisenberg_layers` builds brickwall `GateLayer`s for an even-length chain, with parity `"even"` or `"odd"` mapped to `GateLayer.is_odd=False/True`.
- Order-1 uses repeated `[even, odd]` layers; orders 2 and 4 use specific Suzuki–Trotter/Yoshida sequences with pre-assigned `layer_index` in execution order.
- All Trotter gates in a layer share the same 4x4 local evolution matrix `exp(-i·coeff·H_local)` and act on disjoint nearest-neighbour bonds.

# Tests:
Pytest tests can be found in the pytests/ folder.
The test/ folder contains instead some older tests, and tests that I want to run on the fly for quick checks.

# Optimization Approaches
The repository implements two main strategies for circuit compression, demonstrated in the `experiments/` folder.

## Riemannian Adam on SU(4) Gates
- **Experiment File**: `experiments/riemannian_adam/riemannian_adam_experiment.py`
- **Optimizer Logic**: `rqcopt_mpo/optimization/riemannian_adam/optimizer.py`

This method treats each 2-qubit gate in a standard brickwall circuit as a point on the Stiefel manifold of 4x4 unitary matrices. It performs gradient descent directly on this manifold.

- **Gradient Calculation**: The Euclidean gradient of the loss function with respect to a gate's matrix is computed using the standard "punching a hole" tensor network contraction.
- **Riemannian Update**: The optimization follows these steps for each gate:
  1. The Euclidean gradient is projected onto the tangent space of the manifold at the current gate's matrix.
  2. An Adam-like update rule determines the step direction and magnitude within the tangent space.
  3. The new gate matrix is found by "retracting" from the tangent space back onto the manifold.
- This approach is general and does not assume any specific structure for the 2-qubit gates beyond being unitary.

## Parameter-based Adam on Hardware-Friendly Gates
- **Experiment File**: `experiments/rzz_ising_experiment/rzz_ising_experiment_hw_friendly.py`
- **Optimizer Logic**: `rqcopt_mpo/optimization/rzz_ising_optimizer/rzz_ising_optimizer_hw_friendly.py`

This method is tailored for circuits constructed from a specific, hardware-friendly gate set, typically for simulating the Transverse Field Ising Model (TFIM).

- **Ansatz Structure**: The circuit is built from layers of single-qubit rotations and two-qubit RZZ gates. An initial Trotter circuit is decomposed into a fixed structure where gates are parameterized by angles.
- **Optimization**: The optimization is performed on the scalar parameters (angles) of the gates, not the full gate matrices. A standard Adam optimizer (`rqcopt_mpo/optimization/weyl_optimizer/adam.py`) updates a flat vector containing all circuit parameters.
- **Gradient Calculation**: The gradient is computed in two stages:
  1. The Euclidean gradient with respect to the full gate matrix is found via tensor network contraction.
  2. The chain rule is then applied to obtain the gradient with respect to the underlying parameters. These parameter-gradient functions are registered within the Adam optimizer.
- **Quantum Natural Gradient (QNG)**: This approach also supports QNG. When enabled, it rescales the parameter gradients by the inverse of the Fubini-Study metric tensor, accounting for the geometry of the parameter space to potentially improve convergence.

# Tests:
Pytest tests can be found in the pytests/ folder.
The test/ folder contains instead some older tests, and tests that I want to run on the fly for quick checks.

# Qiskit functions
In case questions concern Functions that are inherited from Qiskit or Pennylane: read the online documentation for Qiskit and Pennylane.

# For now
The dagger method in mpo_dataclass should in principle also switch the bond legs, not just the physical legs, changing the canonicity of the MPO. Applying this correction would require to reverse the list of the mpo and change a bunch of additional core functions that depend on it. For simplicity, we keep the current code, where the bond dimensions are not transposed, and therefore the canonicity is preserved. 
In case later we decide to change also the bond legs to be more accurate in our mathematical picture, here are the affected areas that need change: 
- rqcopt_mpo/mpo/mpo_dataclass.py: dagger() (must reverse list) and to_matrix() (must handle reversed order if called on a daggered MPO).
- rqcopt_mpo/tensor_network/core_ops_helpers.py: hs_inner_product_from_mpo() needs site-index remapping.
- rqcopt_mpo/optimization/optimizer.py: All sweep logic (_layer_pass_left_to_right, etc.) and environment calculations must be updated to map mpo_ref indices back to physical site indices.
- rqcopt_mpo/optimization/gradient.py: compute_gate_environment_tensor would need to pull the correct tensor from the reversed reference.
