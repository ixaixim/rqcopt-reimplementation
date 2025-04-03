import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
from numpy import kron

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def su2_from_euler(theta, phi, lam):
    """
    Constructs an SU(2) matrix from Euler angles.
    Here we use the common R_Z R_Y R_Z parametrization:
    U = exp(-i*(phi+lam)/2) * [[cos(theta/2), -exp(-i*(phi-lam))*sin(theta/2)],
                               [exp(i*(phi-lam))*sin(theta/2), exp(i*(phi+lam)/2)*cos(theta/2]]
    (Up to a global phase, which does not affect the decomposition.)
    """
    return np.array([
        [np.exp(-1j*(phi+lam)/2)*np.cos(theta/2), -np.exp(-1j*(phi-lam)/2)*np.sin(theta/2)],
        [np.exp(1j*(phi-lam)/2)*np.sin(theta/2), np.exp(1j*(phi+lam)/2)*np.cos(theta/2)]
    ], dtype=complex)

def U_d(d):
    """
    Constructs the special two-qubit unitary:
    U_d = exp(-i (d_x X⊗X + d_y Y⊗Y + d_z Z⊗Z))
    where d is a 3-element vector [d_x, d_y, d_z].
    """
    d_x, d_y, d_z = d
    H = d_x * kron(X, X) + d_y * kron(Y, Y) + d_z * kron(Z, Z)
    return expm(-1j * H)

def reconstruct_unitary(params):
    """
    Given a 15-element parameter vector, reconstruct the full two-qubit unitary.
    Parameter breakdown:
      - params[0:3]: Euler angles (theta, phi, lambda) for U_A
      - params[3:6]: Euler angles for U_B
      - params[6:9]: Euler angles for V_A
      - params[9:12]: Euler angles for V_B
      - params[12:15]: parameters (d_x, d_y, d_z) for U_d
    """
    UA_angles = params[0:3]
    UB_angles = params[3:6]
    VA_angles = params[6:9]
    VB_angles = params[9:12]
    d_params   = params[12:15]
    
    UA = su2_from_euler(*UA_angles)
    UB = su2_from_euler(*UB_angles)
    VA = su2_from_euler(*VA_angles)
    VB = su2_from_euler(*VB_angles)
    Ud = U_d(d_params)
    
    return kron(UA, UB) @ Ud @ kron(VA, VB)

def cost_function(params, U_target):
    """
    The cost is the Frobenius norm of the difference between the target unitary
    and the reconstructed unitary.
    """
    U_rec = reconstruct_unitary(params)
    diff = U_target - U_rec
    return np.linalg.norm(diff, ord='fro')

# Example usage
if __name__ == "__main__":
    # For demonstration, let's generate a random target unitary by choosing
    # a random set of parameters. In practice, U_target would be provided.
    np.random.seed(0)
    true_params = np.random.uniform(0, 2*np.pi, 15)
    U_target = reconstruct_unitary(true_params)
    
    # Initial guess for the parameters (could be random or based on some heuristic)
    initial_guess = np.random.uniform(0, 2*np.pi, 15)
    
    # Perform the optimization; note that the problem is non-linear and non-convex.
    opt = {"gtol": 1e-6}
    result = minimize(cost_function, initial_guess, args=(U_target,), method='BFGS', tol=1e-2, options=opt)
    
    if result.success:
        found_params = result.x
        print("Optimized parameters:")
        print(found_params)
        print("Final cost (should be near zero):", cost_function(found_params, U_target))
    else:
        print("Optimization failed:", result.message)
