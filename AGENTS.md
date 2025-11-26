# Environment Configuration
When executing Python scripts, ensure the temporary directory is defined and use the project virtual environment. 
ALWAYS run the command using this exact prefix (only if running from VS Code Extension, NOT from Codex CLI):
If you want to run python:
`export TMPDIR=/tmp && /home/neelmiscia/data/environments/rqcopt/bin/python3`

If you want to run a pytest: 
`export TMPDIR=/tmp /home/neelmiscia/data/environments/rqcopt/bin/pytest`

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

# Qiskit functions
In case questions concern Functions that are inherited from Qiskit or Pennylane: read the online documentation for Qiskit and Pennylane.

