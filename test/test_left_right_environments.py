# create an MPO from a circuit representing the same unitary.
# for a given top/bottom, left/right environment, compute Tr(E G), where E is the environment tensor, G is the respective gate.
# The result should equal Tr(U^dag U) = 2^n, i.e. the full trace.

# test also with arbitrary unitaries. 
# expected behavior: the result should yield the full trace
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

seed_init = 42 
seed_target = 44
n_sites = 4

# generate random target circuit.
target_circuit = generate_random_circuit(
    n_sites=n_sites,
    n_layers=6,
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
init_circuit = target_circuit.copy()
# 
# generate initial circuit (different than target).
# init_circuit = generate_random_circuit(
#     n_sites=n_sites,
#     n_layers=6,
#     p_single=0.3,
#     p_two=0.3,
#     seed=seed_init,
#     gate_name_single='U1',
#     gate_name_two='U2',
#     dtype=jnp.complex128
#     )

# compute and store all top environments
print("     Storing top environments:")
E_top = compute_upper_lower_environments(mpo_ref=target_mpo_dag, circuit=init_circuit, direction='top', init_direction='left_to_right', max_bondim_env=128)
# compute and store all bottom environments
print("     Storing bottom environments:")
id_mpo = get_id_mpo(nsites=n_sites, dtype=jnp.complex128)
E_bot = compute_upper_lower_environments(mpo_ref=id_mpo, circuit=init_circuit, direction='bottom', init_direction='left_to_right', max_bondim_env=128)

# Test that Tr(E^dag G) = Tr(U^dag U) = Tr(I) = 2^N 
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
        
        # build the environment tensor for this gate
        Env = compute_gate_environment_tensor(
            gate_qubits=gate.qubits,
            E_top_layer=E_top[layer_ind],
            E_bottom_layer=E_bot[layer_ind],
            E_left_boundary=E_left[q0],
            E_right_boundary=E_right[q1]
        )

        # contract to get the trace
        trace = compute_trace(Environment=Env, gate_tensor=gate.tensor)
        print(f"      Trace = {trace}\n")
        # matrix trace
        target_matrix_dag = target_mpo_dag.to_matrix()
        init_matrix = circuit_to_mpo(init_circuit).to_matrix()
        matrix_trace = jnp.trace(target_matrix_dag  @ init_matrix)
        print(f"     Matrix Trace = {matrix_trace}\n")
        assert jnp.allclose(trace, matrix_trace, rtol=1e-6, atol=1e-10), f"{trace} != {matrix_trace}"
        print("Check Passed.")
