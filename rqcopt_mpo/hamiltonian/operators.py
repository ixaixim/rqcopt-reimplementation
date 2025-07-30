# rqcopt_mpo/hamiltonian/operators.py
import jax.numpy as jnp

# Define spin-1/2 operators (Pauli matrices)
# Using complex numbers for generality
I = jnp.eye(2, dtype=jnp.complex128)
X = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
Z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

_PAULI = {
    'X': [[0, 1],
          [1, 0]],
    'Y': [[0, -1j],
          [1j,  0]],
    'Z': [[1,  0],
          [0, -1]],
}

def paulis(dtype=jnp.complex64):
    """
    Return a tuple (X, Y, Z), each as a jnp.ndarray with the given dtype.
    """
    return tuple(
        jnp.array(_PAULI[name], dtype=dtype)
        for name in ('X','Y','Z')
    )

def two_qubit_paulis(dtype=jnp.complex64):
    X, Y, Z = paulis(dtype)
    return jnp.kron(X, X), jnp.kron(Y, Y), jnp.kron(Z, Z)


# # You can also define spin ladder operators if needed
# S_plus = 0.5 * (X + 1j * Y)
# S_minus = 0.5 * (X - 1j * Y)
