import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))

from trotter_decomposition import kron_n, first_order_trotter, second_order_trotter, suzuki_trotter
from brickwall_circuit import brickwall_circuit

def embed_two_qubit_operator(op, pos, N):
    """
    Embed a two-qubit operator `op` acting on qubits pos and pos+1
    into an N-qubit Hilbert space.
    """
    left = np.eye(2**pos, dtype=complex)
    right = np.eye(2**(N-pos-2), dtype=complex)
    return np.kron(np.kron(left, op), right)


def plot_trotter_error():
    """
    For a fixed total time t, compute the error between the exact time evolution
    and the Trotterized evolution operators
    """
    # Create a list of Δt values on a log scale.
    delta_t_list = np.logspace(-3, -1, num=30)
    errors = []
    
    # Construct the full Hamiltonian (reference).
    N = 6 
    J = 1.0
    D = 0.5
    h = 0.5
    t = 1

    # Define Pauli matrices and identity
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    

    H_full = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):

        # Group gates based on the parity of the first site index in the bond
        if i % 2 == 0:
          # Construct the two-qubit Hamiltonian for bond (i, i+1)
          # The field term is applied in the even bond.
          H_bond = J * (np.kron(X, X) + np.kron(Y, Y)) + D*np.kron(Z, Z) + h * (np.kron(Z, I) + np.kron(I, Z))
          

        else:
          H_bond = J * (np.kron(X, X) + np.kron(Y, Y)) + D*np.kron(Z, Z) 

        H_full += embed_two_qubit_operator(H_bond, i, N)

    errors_first = []
    # errors_second = []
    # errors_fourth = []
    for delta_t in delta_t_list:
        n = int(t/delta_t)
        # Compute the exact time evolution operator.
        U_exact = expm(-1j * n * delta_t * H_full)
        
        # Compute the Trotterized evolution operators.
        even_terms, odd_terms = brickwall_circuit(N, J, D, h, delta_t)
        odd_terms = odd_terms
        exp_H_even = kron_n(even_terms)
        exp_H_odd = kron_n(odd_terms)

        U_trotter_first = np.linalg.matrix_power(exp_H_even @ exp_H_odd, n)

        # Trotterization adds a phase (since the exponential adds a constant, see Baker Campbell Haudorff formula ), which affects the norm difference. We need to remove it. 
        # Compute the product U_exact^\dagger U_trotter_first
        prod = U_exact.conj().T @ U_trotter_first
        # Extract the global phase from the trace (the overall phase)
        global_phase = np.angle(np.trace(prod))
        # Remove the global phase from U_trotter_first
        U_trotter_aligned = U_trotter_first * np.exp(-1j * global_phase)

        # Compute the error as the spectral norm of the difference.
        error_first = np.linalg.norm(U_exact - U_trotter_aligned, ord=2)
        # error_second = np.linalg.norm(U_exact - U_trotter_second, ord=2)
        # error_fourth = np.linalg.norm(U_exact - U_trotter_fourth, ord=2)
        
        errors_first.append(error_first)
        # errors_second.append(error_second)
        # errors_fourth.append(error_fourth)
    
    # Plotting on a log-log scale.
    plt.figure(figsize=(8, 6))
    plt.loglog(delta_t_list, errors_first, 'o-', label='First-Order Trotter Error')
    # plt.loglog(delta_t_list, errors_second, 's-', label='Second-Order Trotter Error')
    # plt.loglog(delta_t_list, errors_fourth, '^-', label='Fourth-Order Trotter Error')
    
    # Plot reference lines.
    # For first-order splitting, overall error ∼ O(Δt)
    ref_constant_first = errors_first[0] / delta_t_list[0]
    plt.loglog(delta_t_list, ref_constant_first * delta_t_list, '--', label=r'Reference $O(\Delta t)$')
    # For second-order splitting, overall error ∼ O(Δt²)
    # ref_constant_second = errors_second[0] / (delta_t_list[0]**2)
    # plt.loglog(delta_t_list, ref_constant_second * (delta_t_list**2), '--', label=r'Reference $O(\Delta t^2)$')
    # For fourth-order splitting, overall error ∼ O(Δt^4)
    # ref_constant_fourth = errors_fourth[0] / (delta_t_list[0]**4)
    # plt.loglog(delta_t_list, ref_constant_fourth * (delta_t_list**4), '--', label=r'Reference $O(\Delta t^4)$')

    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Spectral norm error')
    plt.title('Scaling of Trotter Error with Time Step')
    plt.legend()
    
    # Set x-limits to cover the Δt values.
    plt.xlim([delta_t_list[0], delta_t_list[-1]])
    
    # Determine y-limits based on the computed errors.
    # all_errors = np.array(errors_first + errors_second + errors_fourth)
    # y_min = all_errors.min() * 0.5  # a bit below the minimum error
    # y_max = all_errors.max() * 2    # a bit above the maximum error
    # plt.ylim([y_min, y_max])
    
    plt.savefig('test/trotter_error.png')


if __name__ == '__main__':
    plot_trotter_error()