import jax
import jax.numpy as jnp
import quimb.tensor as qtn
from autoray import do
import jax_config
from riemannian_adam import RiemannianAdam
from geometry import project_to_tangent

def build_brickwall_circuit_tn(unitaries, n_qubits, n_layers):
    """
    Builds a quimb TensorNetwork for a brickwall circuit.
    unitaries: jax array of shape (G, 2, 2, 2, 2) or (G, 4, 4)
    """
    tn = qtn.TensorNetwork([])
    current_inds = [f'b{i}' for i in range(n_qubits)]

    u_idx = 0
    for l in range(n_layers):
        is_odd = (l % 2 == 1)
        start_q = 1 if is_odd else 0

        new_inds = list(current_inds)
        for q in range(start_q, n_qubits - 1, 2):
            U = unitaries[u_idx]
            u_idx += 1

            out_q_ind = f'l{l}_q{q}'
            out_qp1_ind = f'l{l}_q{q+1}'

            tn.add_tensor(qtn.Tensor(
                data=U.reshape(2, 2, 2, 2),
                inds=(out_q_ind, out_qp1_ind, current_inds[q], current_inds[q+1]),
                tags={f'LAYER_{l}', f'GATE_{q}_{q+1}'}
            ))

            new_inds[q] = out_q_ind
            new_inds[q+1] = out_qp1_ind

        current_inds = new_inds

    for i in range(n_qubits):
        tn.reindex_({current_inds[i]: f't{i}'})

    return tn

def compute_overlap(unitaries, target_mpo_tensors, n_qubits, n_layers):
    circ_tn = build_brickwall_circuit_tn(unitaries, n_qubits, n_layers)

    target_tn = qtn.TensorNetwork([])
    for i, data in enumerate(target_mpo_tensors):
        l_ind = f'm_bond_{i}'
        r_ind = f'm_bond_{i+1}'

        target_tn.add_tensor(qtn.Tensor(
            data=data.conj(),
            inds=(l_ind, f'b{i}', f't{i}', r_ind),
            tags={f'MPO_{i}'}
        ))

    target_tn.reindex_({f'm_bond_0': 'm_bond_closed', f'm_bond_{n_qubits}': 'm_bond_closed'})

    full_tn = circ_tn & target_tn
    # backend is set via autoray when using jax.grad
    return full_tn.contract()

def hst_loss(unitaries, target_mpo_tensors, n_qubits, n_layers, is_normalized=False):
    overlap = compute_overlap(unitaries, target_mpo_tensors, n_qubits, n_layers)
    if is_normalized:
        c = 1.0 / (2 ** n_qubits)
    else:
        c = 1.0 / (2 ** (2 * n_qubits))

    return 1.0 - c * jnp.abs(overlap)**2

def optimize_circuit(target_mpo_tensors, n_qubits, n_layers, n_steps=100, lr=1e-2, is_normalized=False):
    # Initialize random unitaries
    n_gates = 0
    for l in range(n_layers):
        is_odd = (l % 2 == 1)
        start_q = 1 if is_odd else 0
        n_gates += (n_qubits - start_q) // 2

    key = jax.random.PRNGKey(42)
    def random_unitary(key):
        H = jax.random.normal(key, (4, 4), dtype=jnp.complex128)
        H = H + H.conj().T
        return jax.scipy.linalg.expm(1j * H)

    keys = jax.random.split(key, n_gates)
    unitaries = jax.vmap(random_unitary)(keys)

    opt = RiemannianAdam(lr=lr)
    state = opt.init(unitaries)

    @jax.jit
    def step_fn(unitaries, state):
        loss_val, grad = jax.value_and_grad(hst_loss)(unitaries, target_mpo_tensors, n_qubits, n_layers, is_normalized)
        U_next, state_next, stats = opt.step(unitaries, grad, state)
        return U_next, state_next, loss_val

    history = []
    for i in range(n_steps):
        unitaries, state, loss_val = step_fn(unitaries, state)
        history.append(float(loss_val))
        if i % 10 == 0:
            print(f"Step {i}, Loss: {loss_val}")

    return unitaries, history
