# testing: Target MPO: identity, Initial circuit: identical circuit. 
# Expected behavior: Cost is already zero and should never increase. 
import rqcopt_mpo.jax_config

from rqcopt_mpo.mpo.mpo_builder import get_id_mpo
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax 

# parameters
n_sites = 5

# generate id mpo
target = get_id_mpo(5)

# generate identical circuit
I2 = np.eye(2, dtype=jnp.complex128)
I4 = np.eye(4, dtype=jnp.complex128)

# --- Layer 0: two‐qubit on (0,1), (2,3) and single‐qubit on (4) ---
g01 = Gate(matrix=I4, qubits=(0, 1), layer_index=0, name="ID", params=())
g23 = Gate(matrix=I4, qubits=(2, 3), layer_index=0, name="ID", params=())
g4  = Gate(matrix=I2, qubits=(4,),   layer_index=0, name="ID", params=())

layer0 = GateLayer(layer_index=0, is_odd=True,  gates=[g01, g23, g4])

# --- Layer 1: single‐qubit on (1), (3), (4) ---
g1 = Gate(matrix=I2, qubits=(1,), layer_index=1, name="ID", params=())
g3 = Gate(matrix=I2, qubits=(3,), layer_index=1, name="ID", params=())
g4b = Gate(matrix=I2, qubits=(4,), layer_index=1, name="ID", params=())

layer1 = GateLayer(layer_index=1, is_odd=False, gates=[g1, g3, g4b])

circuit = Circuit(n_sites=5, layers=[layer0, layer1])

circuit.print_gates()


# run optimization routine
_, loss = optimize_circuit_local_svd(circuit_initial=circuit, mpo_ref=target, num_sweeps=1, max_bondim_env=64)

loss_hst = 1 - 1/2**(2*n_sites) * np.abs(loss)**2

# plot loss
plt.figure()
plt.plot(loss_hst)
plt.savefig('trace_bottom_up_pass.png')
print(loss_hst)
