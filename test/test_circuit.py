from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
import numpy as np
import jax.numpy as jnp
##############################################
#testing random MPO and random CIRCUIT
# rng = np.random.default_rng(42)
# seed = 42
# qc = generate_random_circuit(
#         n_sites=6,
#         n_layers=4,
#         p_single=0.25,
#         p_two=0.35,
#         rng=rng,
#         seed=seed,
#      )

# qc.print_gates(max_per_layer=6)
##############################################

# Testing Gate copy
matrix = jnp.array([[1,1],[1,-1]], dtype=float) / jnp.sqrt(2)
gate1_jax = Gate(matrix=matrix, qubits=(0,), layer_index=0)
gate1_jax_copy = gate1_jax.copy()
try:
        print(f"\ngate1_jax is gate1_jax_copy: {gate1_jax is gate1_jax_copy}")
        print(f"gate1_jax.matrix is gate1_jax_copy.matrix: {gate1_jax.matrix is gate1_jax_copy.matrix}")
        print(f"JAX Matrices are equal: {jnp.array_equal(gate1_jax.matrix, gate1_jax_copy.matrix)}")
except ImportError:
        print("\nJAX not installed, skipping JAX example.")

# Testing GateLayer copy
# Create some dummy gates
gate_a = Gate(matrix=jnp.array([[1,0],[0,1]]), qubits=(0,), layer_index=0, name="ID0")
gate_b = Gate(matrix=jnp.array([[0,1],[1,0]]), qubits=(1,), layer_index=0, name="X1")

# Create an original layer
original_layer = GateLayer(layer_index=0, is_odd=False, gates=[gate_a, gate_b])

# Create a deep copy of the layer
copied_layer = original_layer.copy()

# --- Verification ---
# 1. Different GateLayer objects
print(f"Original layer is copied layer: {original_layer is copied_layer}") # False

# 2. Different lists of gates
print(f"Original layer.gates is copied_layer.gates: {original_layer.gates is copied_layer.gates}") # False

    # 3. Gates within the list are different objects (due to Gate.copy())
if original_layer.gates and copied_layer.gates:
        print(f"Original layer.gates[0] is copied_layer.gates[0]: {original_layer.gates[0] is copied_layer.gates[0]}") # False
        # And their matrices should be different objects but equal content initially
        print(f"Original gate matrix is copied gate matrix: {original_layer.gates[0].matrix is copied_layer.gates[0].matrix}") # False
        print(f"Matrices content equal: {np.array_equal(original_layer.gates[0].matrix, copied_layer.gates[0].matrix)}") # True


#     # 4. Modifying a gate in the copied layer should not affect the original
        # NOTE: only do with numpy, not jax. 
# if copied_layer.gates:
#         copied_layer.gates[0].name = "MODIFIED_ID0"
#         copied_layer.gates[0].matrix = 99 # Modify the matrix data

#         print(f"\nOriginal layer gate name: {original_layer.gates[0].name if original_layer.gates else 'N/A'}")
#         print(f"Copied layer gate name: {copied_layer.gates[0].name if copied_layer.gates else 'N/A'}")

#         print(f"Original gate matrix[0,0]: {original_layer.gates[0].matrix[0,0] if original_layer.gates else 'N/A'}")
#         print(f"Copied gate matrix[0,0]: {copied_layer.gates[0].matrix[0,0] if copied_layer.gates else 'N/A'}")

# 5. Modifying the list of gates in the copied layer (e.g., appending a new gate)
gate_c = Gate(matrix=np.array([[1,0],[0,-1]]), qubits=(2,), layer_index=0, name="Z2")
copied_layer.gates.append(gate_c)

print(f"\nLength of original_layer.gates: {len(original_layer.gates)}")
print(f"Length of copied_layer.gates: {len(copied_layer.gates)}")

# Testing Circuit copy