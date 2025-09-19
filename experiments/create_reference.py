import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.optimization.utils import overlap_to_loss
import jax.numpy as jnp
import numpy as np

n_sites = 4 # choose even number
J = 1.0
D = -1.0
h = 0
t = 0.5 # time of evolution

reps = 10
order = 4
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False

# set up target MPO
target_circ = trotterized_heisenberg_circuit(    
    n_sites=n_sites, J=J, D=D, h=h,
    order=4, dt=dt, reps=reps,
    dtype=dtype
)
target_matrix = target_circ.to_matrix()

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

print(f"initial circuit layers: {init_circ.num_layers}")

init_matrix = init_circ.to_matrix()
trace = np.trace(target_matrix.conjugate().T @ init_matrix)
hst_cost = overlap_to_loss(trace, n_sites=n_sites, normalize=False)
print(f"Overlap: {trace}")
print(f"HST fidelity: {hst_cost}")


random_seed = 42
random_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=init_circ.num_layers,
    seed=random_seed,
    dtype=dtype,
)

random_matrix = random_circuit.to_matrix()


def _unitarize(matrix: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(matrix)
    diag = np.diag(r)
    phases = np.where(np.abs(diag) > 0, diag / np.abs(diag), 1.0 + 0.0j)
    return (q * phases.reshape(1, -1)).astype(matrix.dtype)


noisy_circuit = random_circuit.copy()
noise_rng = np.random.default_rng(random_seed + 1)

for layer in noisy_circuit.layers:
    for gate in layer.gates:
        perturbation = 1e-3 * (
            noise_rng.normal(size=gate.matrix.shape) + 1j * noise_rng.normal(size=gate.matrix.shape)
        )
        noisy_matrix = np.array(gate.matrix) + perturbation.astype(gate.matrix.dtype)
        gate.matrix = _unitarize(noisy_matrix)

noisy_matrix = noisy_circuit.to_matrix()

pair_trace = np.trace(random_matrix.conjugate().T @ noisy_matrix)
pair_hst = overlap_to_loss(pair_trace, n_sites=n_sites, normalize=False)
print(f"Random vs noisy overlap: {pair_trace}")
print(f"Random vs noisy HST fidelity: {pair_hst}")
