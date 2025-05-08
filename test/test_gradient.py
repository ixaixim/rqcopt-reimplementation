import jax
import numpy as np
import jax.numpy as jnp
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.gradient import compute_gate_environment_tensor

# NOTE: before you run the test. Sometimes it passes, sometimes it does not. Due to Rounding errors. 
# Solution: find h sweet spot, or increase tolerance. 

# Anatomy of float number (can be stored with 32 or 64 bits):
# 1 bit -> sign bit
# 8 or 11 bits-> exponent.
# 23 or 52 bits -> mantissa (fraction)

# the 32 bit float is expressed as (-1)^ sign x (1.fraction) x 2^(exponent-127)
# the mantissa holds the significant digits, e.g. with 23 bits you get log10(2**23) ≈ 6.9  →  about 7 reliable decimal digits

# complex64 means 64 bits per complex entry. half (32) is devoted to the real component, half to the imaginary component. 

# in this example, the norm of the gradient/finite diff is appx. 3000, while the mean absolute difference  0.005 
# when we subtract gradient and finite difference, we therefore expose an appx signal of 0.005

# Two kinds of error compete:
# truncation error of the central difference: grows when h is large, shrink when h gets smaller
# round-off/cancellation error (loss of significant digits): grows when h is small. Originates by subtracting f_plus and f_minus to give zero and then dividing by 2h (wrong derivative!)
def test_environment_finite_difference_2q():
    print("\n--- Testing 2Q Gate Environment via Finite Differences ---")
    n_sites = 5
    phys_dim = 2
    bond_top = 4
    bond_bot = 3
    bond_mid_t = 5 # Bond between E_top[i] and E_top[i+1]
    bond_mid_b = 6 # Bond between E_bot[i] and E_bot[i+1]
    bond_left_t = 2 # Left boundary virtual bonds for E_top
    bond_left_b = 3
    bond_right_t = 4
    bond_right_b = 5
    dtype = jnp.complex64

    # Gate position
    i = 1
    ip1 = 2
    gate_qubits = (i, ip1)

    # Create dummy tensors with *correct matching dimensions*
    # Note: Using random complex numbers
    key = jax.random.PRNGKey(0)
    def rand_complex(shape):
        key_re, key_im = jax.random.split(jax.random.PRNGKey(np.random.randint(0, 10000))) # Vary seed for more randomness
        return (jax.random.normal(key_re, shape, dtype=jnp.float32) +
                1j * jax.random.normal(key_im, shape, dtype=jnp.float32)).astype(dtype)

    # Boundary Environments
    E_left_boundary = rand_complex((bond_left_t, bond_left_b))
    E_right_boundary = rand_complex((bond_right_t, bond_right_b))

    # MPO Tensors for the relevant sites
    E_top_i = rand_complex((bond_left_t, phys_dim, phys_dim, bond_mid_t))
    E_top_ip1 = rand_complex((bond_mid_t, phys_dim, phys_dim, bond_right_t))
    E_bottom_i = rand_complex((bond_left_b, phys_dim, phys_dim, bond_mid_b))
    E_bottom_ip1 = rand_complex((bond_mid_b, phys_dim, phys_dim, bond_right_b))

    # Create dummy MPO objects (only need relevant tensors for the function call)
    # We create full lists just to satisfy the MPO constructor checks
    # Generate dummy_top_tensors with matching bond dimensions
    dummy_top_tensors = []
    for site in range(n_sites):
        if site == 0:
            # Left boundary: left bond = 1, right bond = bond_left_t
            shape = (1, phys_dim, phys_dim, bond_left_t)
        elif site == i:
            # Site i: left bond = bond_left_t, right bond = bond_mid_t
            shape = (bond_left_t, phys_dim, phys_dim, bond_mid_t)
        elif site == ip1:
            # Site i+1: left bond = bond_mid_t, right bond = bond_right_t
            shape = (bond_mid_t, phys_dim, phys_dim, bond_right_t)
        elif site == n_sites - 1:
            # Right boundary: left bond = 1 (matches previous), right bond = 1
            shape = (1, phys_dim, phys_dim, 1)
        else:
            # Filler sites: left bond = bond_right_t, right bond = 1
            shape = (bond_right_t, phys_dim, phys_dim, 1)
        
        dummy_top_tensors.append(np.zeros(shape, dtype=complex))

    # Display the shapes for verification
    print('     Dummy Top MPO Shape:')
    for idx, tensor in enumerate(dummy_top_tensors):
        print(f"Site {idx}: shape {tensor.shape}")

    # Generate dummy_bot_tensors with matching bond dimensions
    dummy_bot_tensors = []
    for site in range(n_sites):
        if site == 0:
            # Left boundary: left bond = 1, right bond = bond_left_b
            shape = (1, phys_dim, phys_dim, bond_left_b)
        elif site == i:
            # Site i: left bond = bond_left_b, right bond = bond_mid_b
            shape = (bond_left_b, phys_dim, phys_dim, bond_mid_b)
        elif site == ip1:
            # Site i+1: left bond = bond_mid_b, right bond = bond_right_b
            shape = (bond_mid_b, phys_dim, phys_dim, bond_right_b)
        elif site == n_sites - 1:
            # Right boundary: left bond = 1, right bond = 1
            shape = (1, phys_dim, phys_dim, 1)
        else:
            # Filler sites: left bond = bond_right_b, right bond = 1
            shape = (bond_right_b, phys_dim, phys_dim, 1)
        
        dummy_bot_tensors.append(np.zeros(shape, dtype=complex))

    # Display the shapes for verification
    print('     Dummy Bottom MPO Shape:')
    for idx, tensor in enumerate(dummy_bot_tensors):
        print(f"Site {idx}: shape {tensor.shape}")



    dummy_top_tensors[i] = E_top_i
    dummy_top_tensors[ip1] = E_top_ip1
    dummy_bot_tensors[i] = E_bottom_i
    dummy_bot_tensors[ip1] = E_bottom_ip1
    # Fill others minimally just to pass MPO checks if needed (not strictly necessary for compute_gate_env call)
    # For simplicity, we'll assume the compute_gate_env function only accesses needed tensors
    E_top_layer = MPO(dummy_top_tensors) # Pass only needed parts
    E_bottom_layer = MPO(dummy_bot_tensors)

    # Random 2Q Gate (Using the target 4D shape)
    G = rand_complex((phys_dim, phys_dim, phys_dim, phys_dim)) # out1, out2, in1, in2

    # --- Analytical Gradient ---
    print("Calculating analytical environment tensor...")
    Env_analytical = compute_gate_environment_tensor(
        gate_qubits, E_top_layer, E_bottom_layer, E_left_boundary, E_right_boundary
    )
    # The gradient w.r.t G is Env.conj()
    Grad_analytical = Env_analytical.conj()
    print(f"Analytical Grad (Env.conj()) shape: {Grad_analytical.shape}")

    # --- Finite Difference Gradient ---
    print("Calculating finite difference gradient...")
    h = 1e-2 # Small step for finite differences
    Grad_fd = jnp.zeros_like(G, dtype=dtype) # Initialize gradient tensor

    # Define the scalar trace function f(gate_tensor)
    def compute_trace(gate_tensor):
        # Contract environment first (same einsum as inside compute_gate_env)
        environment_part = jnp.einsum('ab, acde, efgh, bick, kjfl, hl -> ijdg',
                                   E_left_boundary, E_top_i, E_top_ip1,
                                   E_bottom_i, E_bottom_ip1, E_right_boundary,
                                   optimize='optimal') # same contraction mechanism as in the function that we are testing. 
        # Contract environment with the gate to get the scalar trace
        # For any variational tensor X the scalar cost that you get after contracting the whole network can always be written
        # as a Hilbert-Schmidt inner product L(X) = <E,X> = Tr(E^dag X), where E is the environment tensor obtained by "cutting out" X from the bra-ket network.
        # E has the (out1, out2, in1, in2) ordering.  
        trace_val = jnp.einsum('ijdg, ijdg ->', environment_part.conj(), gate_tensor) # Contracting Env* with G. Most common, easier to read and differentiate. 
        # or equivalently:
        # trace_val = jnp.trace(environment_part.transpose(2,3,0,1).reshape((4,4)).conj() @ gate_tensor.reshape(4,4)) # equivalent, to use if environment is stored with (in1, in2, out1, out2)
        return trace_val

    # Iterate through each element of the gate tensor G
    for p in range(G.shape[0]):
        for q in range(G.shape[1]):
            for r in range(G.shape[2]):
                for s in range(G.shape[3]):
                    # Create unit perturbation tensor at the p,q,r,s index
                    delta = jnp.zeros_like(G, dtype=dtype).at[p, q, r, s].set(1.0)

                    # Calculate f(G + h*delta) and f(G - h*delta)
                    G_plus = G + h * delta
                    G_minus = G - h * delta

                    f_plus = compute_trace(G_plus)
                    f_minus = compute_trace(G_minus)

                    # Compute finite difference derivative for this element
                    deriv = (f_plus - f_minus) / (2 * h)
                    Grad_fd = Grad_fd.at[p, q, r, s].set(deriv)

    print(f"Finite Difference Grad shape: {Grad_fd.shape}")

    # --- Comparison ---
    print("Comparing Analytical and Finite Difference Gradients...")
    is_close = jnp.allclose(Grad_analytical, Grad_fd, rtol=1e-4, atol=1e-6) # where atol: absolute tolerance, rtol: relative tolerance (scales with object)
    max_abs_diff = jnp.max(jnp.abs(Grad_analytical - Grad_fd))
    mean_abs_diff = jnp.mean(jnp.abs(Grad_analytical - Grad_fd))
    norm_analytical = jnp.linalg.norm(Grad_analytical)
    norm_fd = jnp.linalg.norm(Grad_fd)
    relative_diff_norm = jnp.linalg.norm(Grad_analytical - Grad_fd) / max(norm_analytical, norm_fd, 1e-12) # Avoid div by zero

    print(f"Are gradients close (allclose)? {is_close}")
    print(f"Max absolute difference: {max_abs_diff:.2e}")
    print(f"Mean absolute difference: {mean_abs_diff:.2e}")
    print(f"Norm of analytical gradient: {norm_analytical:.2e}")
    print(f"Norm of finite difference gradient: {norm_fd:.2e}")
    print(f"Relative norm difference: {relative_diff_norm:.2e}")

    if not is_close:
        print("!!! Finite difference check FAILED !!!")
    else:
        print("Finite difference check PASSED.")

# --- Run the test ---
if __name__ == "__main__":
    test_environment_finite_difference_2q()
