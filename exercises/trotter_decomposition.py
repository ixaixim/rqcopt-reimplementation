import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


def kron_n(ops):
    """
    Compute the Kronecker product of a list of operators.
    
    Parameters:
    -----------
    ops : list of numpy.ndarray
        List of matrices to tensor together.
    
    Returns:
    --------
    result : numpy.ndarray
        The Kronecker product of the input operators.
    """
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def local_heisenberg_xxz(L, J=1.0, Delta=1.0, periodic=False):
    """
    Construct a list of local Hamiltonian terms for the 1D Heisenberg XXZ model on L sites.
    
    Each local term corresponds to the interaction on a bond between two neighboring sites.
    The Hamiltonian for a bond between sites i and i+1 is given by:
    
        H_{i,i+1} = J * (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Delta * Sz_i Sz_{i+1},
    
    where Sx, Sy, and Sz are the Pauli matrices. For open boundary conditions, bonds are taken
    for i = 0, 1, ..., L-2. If `periodic` is True, an additional term is added for the bond 
    connecting site L-1 and site 0.
    
    Parameters:
    -----------
    L : int
        Number of spins (sites) in the chain.
    J : float, optional
        Coupling constant for the XX (and YY) interactions.
    Delta : float, optional
        Anisotropy parameter for the ZZ interaction.
    periodic : bool, optional
        If True, includes the periodic boundary term connecting site L-1 and site 0.
    
    Returns:
    --------
    local_terms : list of numpy.ndarray
        A list where each element is a matrix acting on the full Hilbert space,
        representing the Heisenberg XXZ interaction on a specific bond.
    """
    # Define the Pauli matrices and the identity.
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # List to store the local Hamiltonian terms.
    local_terms = []
    
    # Loop over all bonds for open boundary conditions.
    for i in range(L - 1):
        # Build the operator for the bond between sites i and i+1.
        # Start by creating a list of operators with the identity at all sites.
        op_list = [I] * L
        
        # Sx_i * Sx_{i+1}
        op_list[i] = sx
        op_list[i+1] = sx
        term_x = kron_n(op_list)
        
        # Reset the operator list for the next term.
        op_list = [I] * L
        # Sy_i * Sy_{i+1}
        op_list[i] = sy
        op_list[i+1] = sy
        term_y = kron_n(op_list)
        
        # Reset the operator list for the next term.
        op_list = [I] * L
        # Sz_i * Sz_{i+1}
        op_list[i] = sz
        op_list[i+1] = sz
        term_z = kron_n(op_list)
        
        # Assemble the local bond Hamiltonian and append it to the list.
        local_terms.append(J * (term_x + term_y) + Delta * term_z)
    
    # Add the periodic boundary term if required.
    if periodic:
        op_list = [I] * L
        op_list[L-1] = sx
        op_list[0]   = sx
        term_x = kron_n(op_list)
        
        op_list = [I] * L
        op_list[L-1] = sy
        op_list[0]   = sy
        term_y = kron_n(op_list)
        
        op_list = [I] * L
        op_list[L-1] = sz
        op_list[0]   = sz
        term_z = kron_n(op_list)
        
        local_terms.append(J * (term_x + term_y) + Delta * term_z)
    
    return local_terms


def first_order_trotter(local_terms, delta_t, n):
    """
    Compute the first-order Trotterized time evolution operator for a Hamiltonian 
    expressed as a sum of local terms
    
    The first-order Trotter step for time δt is defined as:
    
        U_step = ∏_{<i,j>} exp(-i δt H_{ij})
    
    where the product is taken over all bonds (local terms). For a total evolution time 
    T = n * δt, the approximate time evolution operator is given by:
    
        U(T) ≈ (U_step)^n
    
    Parameters:
    -----------
    local_terms : list of numpy.ndarray
        A list of full Hilbert space operators, each corresponding to a local Hamiltonian
        term (bond).
    delta_t : float
        The time step used for each Trotter step.
    n : int
        The number of Trotter steps (so that the total time is T = n * delta_t).
    
    Returns:
    --------
    U : numpy.ndarray
        The approximated time evolution operator for time T:
            U = (∏_{<i,j>} exp(-i δt H_{ij}))^n
    """
    # Compute the evolution operator for a single Trotter step.
    U_step = np.eye(local_terms[0].shape[0], dtype=complex)
    for H_local in local_terms:
        U_step = U_step @ expm(-1j * delta_t * H_local)
    
    # Compute the full time evolution operator by repeating the Trotter step n times.
    U = np.linalg.matrix_power(U_step, n)
    return U


def second_order_trotter(local_terms, delta_t, n):
    """
    Compute the second-order (Suzuki) Trotterized time evolution operator for a Hamiltonian 
    expressed as a sum of local terms.
    
    The Hamiltonian is given by:
        H = sum_gamma H_gamma.
    The second-order (Suzuki) Trotter formula for a time step δt is defined as:
    
        S2(δt) = (∏_{γ in reverse order} exp(-i δt/2 H_γ)) (∏_{γ in forward order} exp(-i δt/2 H_γ)).
    
    For a total evolution time T = n * δt, the approximate time evolution operator is given by:
    
        U(T) ≈ [S2(δt)]^n.
    
    Parameters:
    -----------
    local_terms : list of numpy.ndarray
        A list of local Hamiltonian operators H_gamma, each acting on the full Hilbert space.
    delta_t : float
        The time step used for each Trotter step.
    n : int
        The number of Trotter steps (so that the total time is T = n * delta_t).
    
    Returns:
    --------
    U : numpy.ndarray
        The approximate time evolution operator:
            U = (S2(δt))^n,
        where
            S2(δt) = (∏_{γ in reverse order} exp(-i δt/2 H_γ)) (∏_{γ in forward order} exp(-i δt/2 H_γ)).
    """
    # Dimension of the full Hilbert space is inferred from the first local term.
    dim = local_terms[0].shape[0]
    
    # Compute the forward product: ∏_{γ in forward order} exp(-i δt/2 H_γ)
    U_forward = np.eye(dim, dtype=complex)
    for H_local in local_terms:
        U_forward = U_forward @ expm(-1j * (delta_t/2) * H_local)
    
    # Compute the backward product: ∏_{γ in reverse order} exp(-i δt/2 H_γ)
    U_backward = np.eye(dim, dtype=complex)
    for H_local in reversed(local_terms):
        U_backward = U_backward @ expm(-1j * (delta_t/2) * H_local)
    
    # One full second-order Trotter step.
    U_step = U_backward @ U_forward
    
    # Total time evolution operator approximated by n Trotter steps.
    U = np.linalg.matrix_power(U_step, n)
    return U

def suzuki_trotter(local_terms, k, delta_t, n):
    """
    Compute the S_{2k}(t) product formula using the recursive Suzuki formula.
    
    The recursion is defined as:
    
        S_{2k}(t) = S_{2k-2}(u_k t)^2  S_{2k-2}((1-4u_k)t)  S_{2k-2}(u_k t)^2,
    
    where 
        u_k = 1/(4 - 4^(1/(2k-1))).
    
    The base case is k = 1, corresponding to the second-order Trotter formula S_2(t),
    which is assumed to be implemented by the function second_order_trotter(local_terms, t, n)
    with n = 1 (i.e. a single Trotter step).
    
    Parameters:
    -----------
    local_terms : list of numpy.ndarray
        The list of local Hamiltonian terms.
    k : int
        The recursion level, where k = 1 gives the second-order formula (order 2)
        and k > 1 gives a higher-order (order 2k) formula.
    delta_t : float
        The time step used for each Trotter step.
    n : int
        The number of Trotter steps (so that the total time is T = n * delta_t).

    
    Returns:
    --------
    U : numpy.ndarray
        The product formula approximation S_{2k}(t) as a full evolution operator.
    """
    if k == 1:
        # Base case: return the second-order Trotter formula S_2(t).
        # We assume that second_order_trotter returns S2(t) when n=1.
        return second_order_trotter(local_terms, delta_t, 1)
    else:
        # Compute u_k as given.
        u_k = 1.0 / (4 - 4**(1.0/(2*k - 1)))    
        # Recursively compute S_{2k-2} for the required times.
        S_small = suzuki_trotter(local_terms, k-1, u_k * delta_t, 1)
        S_mid   = suzuki_trotter(local_terms, k-1, (1 - 4 * u_k) * delta_t, 1)
        # Build S_{2k}(t) according to the recursive formula:
        # S_{2k}(t) = [S_{2k-2}(u_k t)]^2  *  S_{2k-2}((1-4u_k)t)  *  [S_{2k-2}(u_k t)]^2
        U_step = S_small @ S_small @ S_mid @ S_small @ S_small
        U = np.linalg.matrix_power(U_step, n)
        return U


def H_from_local(local_terms):
    """
    Construct the full Hamiltonian by summing the list of local Hamiltonians.
    
    Parameters:
    -----------
    local_terms : list of numpy.ndarray
         A list of local Hamiltonian operators acting on the full Hilbert space.
         
    Returns:
    --------
    H : numpy.ndarray
         The full Hamiltonian obtained as the sum of the local terms.
    """
    return np.sum(local_terms, axis=0)

def plot_trotter_error(local_terms, t=1.0):
    """
    For a fixed total time t, compute the error between the exact time evolution
    and the Trotterized evolution operators (both first- and second-order) as a function
    of the time step Δt. Also plot reference curves that illustrate the expected scaling:
    
      - First-order Trotter error: overall error ~ O(Δt)
      - Second-order Trotter error: overall error ~ O(Δt²)
        
    Parameters:
    -----------
    local_terms : list of numpy.ndarray
         A list of local Hamiltonian operators that, when summed, yield the full Hamiltonian.
    t : float
         Total evolution time.
    """
    # Create a list of Δt values on a log scale.
    delta_t_list = np.logspace(-3, -1, num=30)
    errors_first = []   # First-order Trotter errors.
    errors_second = []  # Second-order Trotter errors.
    errors_fourth = []  # Fourth-order Trotter errors.
    
    # Construct the full Hamiltonian (reference).
    H_full = H_from_local(local_terms)
    
    for delta_t in delta_t_list:
        # Compute the number of Trotter steps so that n * delta_t = t.
        n = int(np.round(t / delta_t))
        
        # Compute the exact time evolution operator.
        U_exact = expm(-1j * n * delta_t * H_full)
        
        # Compute the Trotterized evolution operators.
        U_trotter_first = first_order_trotter(local_terms, delta_t, n)
        U_trotter_second = second_order_trotter(local_terms, delta_t, n)
        U_trotter_fourth = suzuki_trotter(local_terms, 2, delta_t, n)
        
        # Compute the error as the spectral norm of the difference.
        error_first = np.linalg.norm(U_exact - U_trotter_first, ord=2)
        error_second = np.linalg.norm(U_exact - U_trotter_second, ord=2)
        error_fourth = np.linalg.norm(U_exact - U_trotter_fourth, ord=2)
        
        errors_first.append(error_first)
        errors_second.append(error_second)
        errors_fourth.append(error_fourth)
    
    # Plotting on a log-log scale.
    plt.figure(figsize=(8, 6))
    plt.loglog(delta_t_list, errors_first, 'o-', label='First-Order Trotter Error')
    plt.loglog(delta_t_list, errors_second, 's-', label='Second-Order Trotter Error')
    plt.loglog(delta_t_list, errors_fourth, '^-', label='Fourth-Order Trotter Error')
    
    # Plot reference lines.
    # For first-order splitting, overall error ∼ O(Δt)
    ref_constant_first = errors_first[0] / delta_t_list[0]
    plt.loglog(delta_t_list, ref_constant_first * delta_t_list, '--', label=r'Reference $O(\Delta t)$')
    # For second-order splitting, overall error ∼ O(Δt²)
    ref_constant_second = errors_second[0] / (delta_t_list[0]**2)
    plt.loglog(delta_t_list, ref_constant_second * (delta_t_list**2), '--', label=r'Reference $O(\Delta t^2)$')
    # For fourth-order splitting, overall error ∼ O(Δt^4)
    ref_constant_fourth = errors_fourth[0] / (delta_t_list[0]**4)
    plt.loglog(delta_t_list, ref_constant_fourth * (delta_t_list**4), '--', label=r'Reference $O(\Delta t^4)$')

    plt.xlabel(r'$\Delta t$')
    plt.ylabel('Spectral norm error')
    plt.title('Scaling of Trotter Error with Time Step')
    plt.legend()
    
    # Set x-limits to cover the Δt values.
    plt.xlim([delta_t_list[0], delta_t_list[-1]])
    
    # Determine y-limits based on the computed errors.
    all_errors = np.array(errors_first + errors_second + errors_fourth)
    y_min = all_errors.min() * 0.5  # a bit below the minimum error
    y_max = all_errors.max() * 2    # a bit above the maximum error
    plt.ylim([y_min, y_max])
    
    plt.savefig('plots/trotter_error.png')

