import jax
import jax.numpy as jnp
import quimb.tensor as qtn
from rqcopt import optimize_circuit

def test_optimization():
    n_qubits = 4
    n_layers = 2

    # Create a target MPO: Identity
    target_mpo_tensors = []
    for i in range(n_qubits):
        # (left, out, in, right)
        t = jnp.eye(2, dtype=jnp.complex128).reshape(1, 2, 2, 1)
        target_mpo_tensors.append(t)

    print("Starting optimization...")
    unitaries, history = optimize_circuit(target_mpo_tensors, n_qubits, n_layers, n_steps=50, lr=1e-2)

    print("Initial Loss:", history[0])
    print("Final Loss:", history[-1])

    assert history[-1] < history[0], "Loss did not decrease"
    print("Test passed: Loss decreased from {} to {}".format(history[0], history[-1]))

if __name__ == "__main__":
    test_optimization()
