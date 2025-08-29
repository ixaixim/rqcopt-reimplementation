import rqcopt_mpo.jax_config
from pathlib import Path
from experiments.utils import save_data_npz

import jax 
import jax.numpy as jnp
import numpy as np

# create a simple circuit with 4 qubits and 3 gates. On (0,1), (2,3) on first layer, and (1,2) on second layer.
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.circuit.circuit_builder import _random_unitary
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit
from rqcopt_mpo.circuit.decompose.single_q_decompose import euler_zyz_decompose_circuit 
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo

# optimization: 
from rqcopt_mpo.optimization.parametrized_adam.optimizer import optimize
from scipy.linalg import polar

def make_unitary_128(M) -> np.ndarray:
    """
    Ensure M is a NumPy complex128, unitary matrix via polar decomposition.
    Returns the unitary factor U from M = U H.
    """
    M64 = np.asarray(M, dtype=np.complex128)
    # U, _ = polar(M64)  # U is unitary, _ is Hermitian PSD
    #debug: 
    U = M64
    return U


seed = 42
k0, k1, k2 = jax.random.split(jax.random.PRNGKey(seed), 3)

# # Make three 2-qubit unitaries (4x4), dtype matches the Circuit below
U01 = _random_unitary(4, key=k0, dtype=jnp.complex128)  # for qubits (0,1) on layer 0
# U23 = _random_unitary(4, key=k1, dtype=jnp.complex128)  # for qubits (2,3) on layer 0
# U12 = _random_unitary(4, key=k2, dtype=jnp.complex128)  # for qubits (1,2) on layer 1

U01 = make_unitary_128(U01)  # now np.complex128, unitary
# U23 = make_unitary_128(U23)
# U12 = make_unitary_128(U12)

# # Create Gate instances
g01 = Gate(matrix=U01, qubits=(0, 1), layer_index=0, name="U2")
# # g23 = Gate(matrix=U23, qubits=(2, 3), layer_index=0, name="U2")
# # g12 = Gate(matrix=U12, qubits=(1, 2), layer_index=1, name="U2")

#########################
# debug:
U0 = _random_unitary(2, key=k0, dtype=jnp.complex128)  # for qubits (0,1) on layer 0
U0 = make_unitary_128(U0)
g0 = Gate(matrix=U0, qubits=(0,), layer_index=0, name="U0")
layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g01])

# layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g01])
circ = Circuit(n_sites=4, dtype=jnp.complex128, layers=[layer0])

# Decompose the single qubit gates (num_layers = 7*num_layers)
circ = weyl_decompose_circuit(circ, keep_global_phase=False)
circ_single_qubit_decomp = euler_zyz_decompose_circuit(circ, include_global_phase=True)
circ_single_qubit_decomp.print_gates()

##########################
# Debug
# Decompose the circuit (num_layers = 3*num_layers)
# circ_decomp = weyl_decompose_circuit(circ)

# Decompose the single qubit gates (num_layers = 7*num_layers)
# circ_single_qubit_decomp = euler_zyz_decompose_circuit(circ_decomp, include_global_phase=True)
# circ_single_qubit_decomp.print_gates()

# A = circ.to_matrix()
# B = circ_single_qubit_decomp.to_matrix()

# num = np.vdot(A, B)
# if np.abs(num) < 1e-16:
#     B_aligned = B
# else: 
#     B_aligned = np.exp(-1j*np.angle(num))*B

# try:
#     np.testing.assert_allclose(A, B_aligned, atol=1e-14, rtol=0)
# except AssertionError as e:
#     D = A - B
#     print("FAILED up to global phase.")
#     print("Estimated global phase phi (radians):", np.angle(num))
#     print("max|A - e^{iφ}B| =", np.max(np.abs(D)))
#     # Optional: relative unitary error (Frobenius norm)
#     rel_err = np.linalg.norm(D) / (np.linalg.norm(A) + 1e-16)
#     print("Relative Frobenius error:", rel_err)
#     # Show a small slice to avoid huge dumps
#     print("Top-left 4x4 of diff:\n", D[:4,:4])
#     raise

# print(circuit_to_mpo(A).dagger().to_matrix() @ B_aligned)
############################

# create a target circuit:
U1 = _random_unitary(2, key=k1, dtype=jnp.complex128)  # for qubit (1) on layer 0
U1 = make_unitary_128(U1)
g1 = Gate(matrix=U1, qubits=(0,), layer_index=0, name="U1")
layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g1])

circ_target = Circuit(n_sites=4, dtype=jnp.complex128, layers=[layer0])

circ_target_mpo = circuit_to_mpo(circ_target)
normalize_target = False # it performs poorly if normalized.

circ_target_mpo.left_canonicalize(normalize=normalize_target)
 
lr = 1e-1
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
max_steps = 200
max_bondim_env = 128
svd_cutoff = 0.0

loss = optimize(circ_single_qubit_decomp, circ_target_mpo,
        lr=lr, betas=betas, eps=eps,
        clip_grad_norm=clip_grad_norm, 
        max_steps=max_steps,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff)

base_dir = here = Path(__file__).resolve().parent
# save_data_npz(base_dir, 'loss_riemannian', loss, method='Riemannian_Adam_trotterized_init')


