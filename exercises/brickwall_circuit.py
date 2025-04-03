import numpy as np
from scipy.linalg import expm

# TODO: this brickwall circuit actually just separates in even and odd layer. 
#       need to change name to this function and implement a brickwall circuit which implements a Trotterization step (with repetitions). 
def brickwall_circuit(N, J, D, h, dt):
    """
    Construct a quantum circuit for a 1D transverse Heisenberg XXZ model with a brickwall pattern.
    
    For each nearest-neighbor bond (i, i+1), the Hamiltonian is given by:
      H_bond = J*(X⊗X + Y⊗Y)+ D*Z⊗Z + h*(Z⊗I)
    where the magnetic field term h is applied to the first qubit in the bond.
    
    The time evolution gate for each bond is then computed as:
      U = exp(-i * dt * H_bond)
    
    The gates are grouped into even and odd bonds based on the index i.
    """
    # Define Pauli matrices and identity
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)


    H_even = []
    H_odd = []
    for i in range(N - 1):

      # Group gates based on the parity of the first site index in the bond
      if i % 2 == 0:
        # Construct the two-qubit Hamiltonian for bond (i, i+1)
        # The field term is applied in the even bond.
        H_bond = J * (np.kron(X, X) + np.kron(Y, Y)) + D*np.kron(Z, Z) + h * (np.kron(Z, I) + np.kron(I, Z))
        # U = expm(-1j * dt * H_bond)
        H_even.append(H_bond)

      else:
        H_bond = J * (np.kron(X, X) + np.kron(Y, Y)) + D*np.kron(Z, Z) 
        # U = expm(-1j * dt * H_bond)
        H_odd.append(H_bond)
    H_odd = [np.eye(2, dtype=complex)] + H_odd + [np.eye(2, dtype=complex)]

    even_gates = [expm(-1j * dt * bond) for bond in H_even]
    odd_gates = [expm(-1j * dt * bond) for bond in H_odd]

    # NOTE: the single qubit gates in H_odd are simply adding a phase to the operator

    return even_gates, odd_gates
