# the environment tensor is defined as E^* := d/dG Tr(U_reference^adj U_QC) = d/dG Tr(E^adj G). 
# check that the analytical environment corresponds to the numerical Wirtinger derivative above.
import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_builder import _random_unitary, generate_random_circuit
from rqcopt_mpo.circuit.circuit_dataclasses import Gate
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo, get_id_mpo
from rqcopt_mpo.optimization.optimizer import optimize_circuit_local_svd
from rqcopt_mpo.optimization.gradient import compute_upper_lower_environments, compute_layer_boundary_environments, compute_gate_environment_tensor, compute_trace
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax 


# --------------------------------------------------------------------
# 1. 4-point centred finite difference for the Wirtinger derivative
#    ∂f/∂G
# --------------------------------------------------------------------
def numeric_env(f, G0, h: float = 1e-6):
    """
    Finite-difference approximation of the Wirtinger gradient
    E_num ≈ ∂f / ∂G evaluated at G = G0.

    Parameters
    ----------
    f  : callable(G) -> complex
        Scalar function of the gate tensor.
    G0 : jnp.ndarray  (shape (2,2,2,2) here)
        Gate tensor at which the environment is evaluated.
    h  : float
        Finite-difference step.

    Returns
    -------
    E_num : jnp.ndarray, same shape as G0
        Numerical environment tensor.
    """
    E_num = jnp.zeros_like(G0, dtype=G0.dtype)

    # iterate over all indices of the 4-index tensor
    for idx in np.ndindex(*G0.shape): # (0,0), (0,1), (0,2), etc.
        delta = jnp.zeros_like(G0).at[idx].set(1.0)

        fp  = f(G0 +      h * delta)      # G +  h
        fm  = f(G0 -      h * delta)      # G -  h
        fip = f(G0 + 1j * h * delta)      # G + i h
        fim = f(G0 - 1j * h * delta)      # G - i h

        deriv = ((fp - fm) - 1j * (fip - fim)) / (4.0 * h) # finite difference of real and imaginary part combined.
        E_num = E_num.at[idx].set(deriv)

    return E_num

def numeric_env_conj(f, G0, h: float = 1e-6):
    """
    Finite-difference approximation of the Wirtinger gradient
    E_num ≈ ∂f / ∂G* evaluated at G = G0.

    Parameters
    ----------
    f  : callable(G) -> complex
        Scalar function of the gate tensor.
    G0 : jnp.ndarray  (shape (2,2,2,2) here)
        Gate tensor at which the environment is evaluated.
    h  : float
        Finite-difference step.

    Returns
    -------
    E_num : jnp.ndarray, same shape as G0
        Numerical environment tensor.
    """
    E_num = jnp.zeros_like(G0, dtype=G0.dtype)

    # iterate over all indices of the 4-index tensor
    for idx in np.ndindex(*G0.shape): # (0,0), (0,1), (0,2), etc.
        delta = jnp.zeros_like(G0).at[idx].set(1.0)

        fp  = f(G0 +      h * delta)      # G +  h
        fm  = f(G0 -      h * delta)      # G -  h
        fip = f(G0 + 1j * h * delta)      # G + i h
        fim = f(G0 - 1j * h * delta)      # G - i h

        deriv = ((fp - fm) + 1j * (fip - fim)) / (4.0 * h) # finite difference of real and imaginary part combined.
        E_num = E_num.at[idx].set(deriv)
    return E_num



seed_init = 42 
seed_target = 44
n_sites = 5

# generate random target circuit.
target_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=4,
    p_single=0.3,
    p_two=0.3,
    seed=seed_target,
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )

target_circuit.sort_layers()
target_circuit.print_gates()
# transform target into MPO
print("Converting target circuit to MPO...")
target_mpo = circuit_to_mpo(target_circuit)
target_mpo.left_canonicalize()
target_mpo_dag = target_mpo.dagger()

# generate initial circuit (same as target)
# init_circuit = target_circuit.copy()
# 
# generate initial circuit (different than target).
init_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=10,
    p_single=0.3,
    p_two=0.3,
    seed=seed_init,
    gate_name_single='U1',
    gate_name_two='U2',
    dtype=jnp.complex128
    )
init_circuit.print_gates()
# compute and store all top environments
print("     Storing top environments:")
E_top = compute_upper_lower_environments(mpo_ref=target_mpo_dag, circuit=init_circuit, direction='top', init_direction='left_to_right', max_bondim_env=128)
# compute and store all bottom environments
print("     Storing bottom environments:")
id_mpo = get_id_mpo(nsites=n_sites, dtype=jnp.complex128)
E_bot = compute_upper_lower_environments(mpo_ref=id_mpo, circuit=init_circuit, direction='bottom', init_direction='left_to_right', max_bondim_env=128)

# compute the left and right environment for each gate
for layer_ind in range(init_circuit.num_layers):
    layer = init_circuit.layers[layer_ind]
    print(f"\n===== Layer {layer_ind} =====")

    # compute and store all right environments
    print(f"  Storing right environments for layer index {layer_ind}:")
    E_right = compute_layer_boundary_environments(
        E_top=E_top[layer_ind],
        E_bottom=E_bot[layer_ind],
        layer=layer,
        side='right'
    )

    # compute and store all left environments
    print(f"  Storing left environments for layer index {layer_ind}:")
    E_left = compute_layer_boundary_environments(
        E_top=E_top[layer_ind],
        E_bottom=E_bot[layer_ind],
        layer=layer,
        side='left'
    )
    # now loop over every gate in this layer
    for gate_idx, gate in enumerate(layer.gates):
        print(f"    Gate {gate_idx} on qubits {gate.qubits}:")
        if len(gate.qubits) == 2:
            q0, q1 = gate.qubits  # assuming two-qubit gates; for single-qubit adjust accordingly
        elif len(gate.qubits) == 1:
            q0 = q1 = gate.qubits[0]
        else: 
            raise ValueError("Unsupported number of qubits in gate.")
        
        # build the environment tensor for this gate: analytical derivative
        E_anal = compute_gate_environment_tensor(
            gate_qubits=gate.qubits,
            E_top_layer=E_top[layer_ind],
            E_bottom_layer=E_bot[layer_ind],
            E_left_boundary=E_left[q0],
            E_right_boundary=E_right[q1]
        ).conj()

        if len(gate.qubits) == 2:
            E_anal = E_anal.reshape(4,4)
        # else:
        #     E_anal = jnp.transpose(E_anal, (1,0))
        # compute the numerical derivative
        def f(displaced_gate):
            # replaces circuit gate at the current position with displaced_gate. 
            # displaced_gate differs by gate by a displacement (see numeric_env)
            # calculates the final trace Tr(W^ref U_circuit)
            temp_circuit = init_circuit.copy()
            new_gate = temp_circuit.layers[layer_ind].gates[gate_idx].copy()
            new_gate.matrix = displaced_gate
            temp_circuit.layers[layer_ind].gates[gate_idx] = new_gate
            init_matrix = temp_circuit.to_matrix()# circuit_to_mpo(temp_circuit).to_matrix()
            return np.trace(target_mpo_dag.to_matrix() @ init_matrix)
        # contract to get the trace
        G0 = gate.matrix
        E_num  = numeric_env(f, G0, h=1e-6)
        E_num_conj = numeric_env_conj(f, G0, h=1e-6)

        # test holomorphic assumption
        print("Holomorphic check: d/dz^conj f = 0")
        if not jnp.allclose(E_num_conj, 0.0, rtol=1e-4, atol=1e-8):
            raise AssertionError(
                f"Holomorphic check failed for gate {gate_idx} in layer {layer_ind}: "
                f"max |d/dz^conj f| = {jnp.max(jnp.abs(E_num_conj)):.2e}"
            )
        else:
            print("        ✓ holomorphic check passed")
            
        # test
        if not jnp.allclose(E_anal, E_num, rtol=1e-4, atol=1e-8):
            raise AssertionError(
                f"Mismatch for gate {gate_idx} in layer {layer_ind}: "
                f"max |Δ| = {jnp.max(jnp.abs(E_anal - E_num)):.2e}"
            )
        else:
            print("        ✓ environment check passed")
        


