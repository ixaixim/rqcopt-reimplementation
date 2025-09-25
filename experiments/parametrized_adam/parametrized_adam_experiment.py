import rqcopt_mpo.jax_config
from pathlib import Path
from experiments.utils import save_data_npz

import jax 
import jax.numpy as jnp
import numpy as np

# circuit and MPO:
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo

# decomposition:
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit
from rqcopt_mpo.circuit.decompose.single_q_decompose import euler_zyz_decompose_circuit 

# optimization: 
from rqcopt_mpo.optimization.parametrized_adam.optimizer import optimize

# trotterization:
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit

# debug
from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.optimization.utils import overlap_to_loss

# trotterization params
n_sites = 6 # choose even number
J = 1.0
D = -1.0
h = 0
t = 0.5 # time of evolution

reps = 10
order = 4
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False

# # set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")

target_mpo = circuit_to_mpo(target_circ)
target_mpo.left_canonicalize(normalize=target_is_normalized)

# set up quantum circuit

reps = 3
dt = t/reps
init_circ = trotterized_heisenberg_circuit(
    n_sites=n_sites,
    J=J,
    D=D,
    dt=dt,
    reps=reps,
    order=2,
    dtype=jnp.complex128,
)

print(f"Initial circuit with {init_circ.num_layers} layers")
print(f"Initial Fidelity of Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# weyl decomp
init_circ = weyl_decompose_circuit(init_circ, keep_global_phase=False)
# euler rotation decomp
init_circ = euler_zyz_decompose_circuit(init_circ, include_global_phase=False)
print(f"Init Circuit Decomposed with {init_circ.num_layers} layers")
print(f"Initial Fidelity of Decomposed Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")

# random_seed = 42
# num_layers = 5
# random_circuit = generate_random_circuit(
#     n_sites=n_sites,
#     n_layers=num_layers,
#     seed=random_seed,
#     dtype=dtype,
# )

# random_matrix = random_circuit.to_matrix()


# def _unitarize(matrix: np.ndarray) -> np.ndarray:
#     q, r = np.linalg.qr(matrix)
#     diag = np.diag(r)
#     phases = np.where(np.abs(diag) > 0, diag / np.abs(diag), 1.0 + 0.0j)
#     return (q * phases.reshape(1, -1)).astype(matrix.dtype)


# noisy_circuit = random_circuit.copy()
# noise_rng = np.random.default_rng(random_seed + 1)

# for layer in noisy_circuit.layers:
#     for gate in layer.gates:
#         perturbation = 1e-2 * (
#             noise_rng.normal(size=gate.matrix.shape) + 1j * noise_rng.normal(size=gate.matrix.shape)
#         )
#         noisy_matrix = np.array(gate.matrix) + perturbation.astype(gate.matrix.dtype)
#         gate.matrix = _unitarize(noisy_matrix)

# noisy_matrix = noisy_circuit.to_matrix()

# pair_trace = np.trace(random_matrix.conjugate().T @ noisy_matrix)
# pair_hst = overlap_to_loss(pair_trace, n_sites=n_sites, normalize=False)
# print(f"Random vs noisy overlap: {pair_trace}")
# print(f"Random vs noisy HST fidelity: {pair_hst}")
# init_circ = noisy_circuit
# # set up target MPO
# target_circ = random_circuit
# target_mpo = circuit_to_mpo(target_circ)
# target_mpo.left_canonicalize(normalize=target_is_normalized)
# # weyl decomp
# init_circ = weyl_decompose_circuit(init_circ, keep_global_phase=False)
# # euler rotation decomp
# init_circ = euler_zyz_decompose_circuit(init_circ, include_global_phase=False)
# init_circ.print_gates()

# optimization parameters 
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-8
clip_grad_norm = None
bias_correction = True
max_steps = 100
max_bondim_env = 128
svd_cutoff = 0.0

loss = optimize(init_circ, target_mpo,
        lr=lr, betas=betas, eps=eps,
        clip_grad_norm=clip_grad_norm, 
        bias_correction=bias_correction,
        max_steps=max_steps,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff)

base_dir = here = Path(__file__).resolve().parent
save_data_npz(base_dir, 'loss_parametrized_adam', loss, method='Parametrized_Adam')


