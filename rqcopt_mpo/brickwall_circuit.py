from functools import reduce
from typing import List, Tuple, Optional, Any, Dict 

import jax.numpy as jnp 
from jax.numpy import eye, kron, asarray
from jax import config as c
c.update("jax_enable_x64", True)

# from .fermionic_systems import get_swap_network_trotter_gates_fermi_hubbard_1d, get_swap_network_trotter_gates_molecular
from .spin_systems import get_brickwall_trotter_gates_spin_chain
from .util import get_identity_layers
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, GateLayer, Gate
from qiskit.synthesis import TwoQubitWeylDecomposition

# Define some operators
I = eye(2)
X = asarray([[0., 1.],[1., 0.]])
Y = asarray([[0., -1.j],[1.j, 0.]])
Z = asarray([[1., 0.],[0.,-1.]])
XX = kron(X,X)
YY = kron(Y,Y)
ZZ = kron(Z,Z)
XI = kron(X,I)
IX = kron(I,X)
ZI = kron(Z,I)
IZ = kron(I,Z)


def tensor_product(operators):
    return reduce(kron, operators)

def get_nlayers(degree, n_repetitions, n_orbitals=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    if hamiltonian=='fermi-hubbard-1d':
        # if degree==1:
        #     n_SN_layers = 4*n_repetitions
        # elif degree==2:
        #     n_SN_layers = 3*degree*n_repetitions  # Number of layers in pure swap network
        #     n_SN_layers -= (degree*n_repetitions-1)  # If we absorb 
        # elif degree==4:
        #     n_SN_layers = 20*n_repetitions+1
        pass
    elif hamiltonian=='molecular':
        # assert(type(n_orbitals) is not type(None))
        # if degree in [1,2]:
        #     n_SN_layers = 2*n_orbitals*n_repetitions  # Number of layers in pure swap network
        #     n_SN_layers -= (2*n_repetitions-1)  # If we absorb 
        # elif degree==4:
        #     n_SN_layers = 10*(n_orbitals-1)*n_repetitions+1
        pass
    elif hamiltonian in ['ising-1d','heisenberg']:
        if degree==1:
            n_SN_layers = 2*n_repetitions
        elif degree==2: 
            n_SN_layers = 2*n_repetitions+1 # since the symmetry of the splitting allows to chain the levels together. 
        elif degree==4:
            n_SN_layers = 10*n_repetitions+1
    return n_SN_layers+n_id_layers

def get_initial_gates(n_sites, t, n_repetitions=1, degree=2, 
                      hamiltonian='fermi-hubbard-1d', n_id_layers=0, use_TN=True, **kwargs):
    if hamiltonian == 'fermi-hubbard-1d':
        # T, V = kwargs['T'], kwargs['V']
        # Vlist_start = get_swap_network_trotter_gates_fermi_hubbard_1d(T, V, t, n_sites, n_repetitions, degree, use_TN)
        # first_layer_odd=False
        pass
    elif hamiltonian=='molecular':
    #     T, V = kwargs['T'], kwargs['V']
    #     Vlist_start = get_swap_network_trotter_gates_molecular(T, V, t, n_sites, degree, n_repetitions, use_TN=use_TN)
    #     assert(n_id_layers==0)
        pass
    elif hamiltonian in ['ising-1d', 'heisenberg']:
        Vlist_start = get_brickwall_trotter_gates_spin_chain(t, n_sites, n_repetitions, degree, hamiltonian, use_TN, **kwargs)
        if degree==1: first_layer_odd=True
        elif degree in [2,4]: first_layer_odd=False

    if n_id_layers>0:
        Vlist_start = list(Vlist_start)+list(get_identity_layers(n_sites, n_id_layers, first_layer_odd, use_TN))
    return asarray(Vlist_start)

def get_gates_per_layer(Vlist, n_sites, degree=None, n_repetitions=None,
                        n_layers=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    N_odd_gates, N_even_gates = int(n_sites/2), int(n_sites/2)  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
    if type(n_layers) is type(None):
        assert(type(degree) is not None and type(n_repetitions) is not None)
        n_SN_layers = get_nlayers(degree, n_repetitions, n_sites, n_id_layers, hamiltonian)
    else:
        n_SN_layers = n_layers
        
    if hamiltonian=='fermi-hubbard-1d':
        # odd = False  # First layer is even
        # lim1, lim2 = 0, N_even_gates
        pass
    elif hamiltonian in ['molecular', 'ising-1d', 'heisenberg']:
        odd = True  # First layer is odd
        lim1, lim2 = 0, N_odd_gates

    gates_per_layer, layer_is_odd = [], []
    for _ in range(1, n_SN_layers+1):
        layer_is_odd.append(odd)
        gates_per_layer.append(Vlist[lim1:lim2])
        lim1=lim2; lim2+=N_even_gates if odd else N_odd_gates
        odd = not odd  # Parity of next layer
            
    return gates_per_layer, layer_is_odd

# def circuit_Weyl_decomposition(Vlist, n_sites, degree=None, n_repetitions=None,
#                         n_layers=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
#         gates_per_layer, layer_is_odd = get_gates_per_layer(Vlist, n_sites, degree=degree, n_repetitions=n_repetitions, n_layers=n_layers, n_id_layers=n_id_layers, hamiltonian=hamiltonian)

#     # loop through gates_per_layer
#         # for each gate in layer:
#             # do weyl decomposition of gate
#             # store the gate 
        
# def get_circuit_structure(n_sites, raw_gates_per_layer, layer_is_odd, hamiltonian):
    # circuit = Circuit(n_sites=n_sites, hamiltonian_type=hamiltonian)
    # current_qubit_index = 0 # Keep track of which qubits the gates apply to
    # processed_gates_in_layer = 0

    # for i, (raw_layer_tensors, is_odd_layer) in enumerate(zip(raw_gates_per_layer, layer_is_odd)):
    #     layer_index = i 
    #     gate_layer = GateLayer(layer_index=layer_index, is_odd=is_odd_layer)

    #     # Determine the qubit pairs for this layer based on parity and n_sites
    #     if is_odd_layer:
    #         qubit_pairs = [(k, k + 1) for k in range(0, n_sites -1 , 2)] # (0,1), (2,3), (4,5), ...
    #     else: # Even layer
    #         qubit_pairs = [(k, k + 1) for k in range(1, n_sites -1 , 2)] # (1,2), (3,4), ...

    #     if len(raw_layer_tensors) != len(qubit_pairs):
    #         # Add error handling or logging - something might be wrong
    #         print(f"Warning: Layer {layer_index} mismatch. Expected {len(qubit_pairs)} pairs, found {len(raw_layer_tensors)} tensors.")

    #     for original_tensor, qubits in zip(raw_layer_tensors, qubit_pairs):
    #         # --- Perform Weyl Decomposition ---
    #         # K1l_t, K1r_t, Exp_t, K2l_t, K2r_t, params_a_b_c = perform_weyl_decomposition(original_tensor) 
    #         # Placeholder values - replace with your actual decomposition call
    #         K1l_t, K1r_t, Exp_t, K2l_t, K2r_t = eye(2), eye(2), eye(4), eye(2), eye(2) 
    #         params_a_b_c = (0.0, 0.0, 0.0)
    #         q1, q2 = qubits

    #                 # Create the 5 Gate objects
    #         gate_K2l = Gate(tensor=K2l_t, qubits=(q1,), layer_index=layer_index, 
    #                         name="K2l", original_gate_qubits=qubits, decomposition_part="K2l")
    #         gate_K2r = Gate(tensor=K2r_t, qubits=(q2,), layer_index=layer_index, 
    #                         name="K2r", original_gate_qubits=qubits, decomposition_part="K2r")
    #         gate_Exp = Gate(tensor=Exp_t, qubits=qubits, layer_index=layer_index, 
    #                         name="ExpXYZ", params=params_a_b_c, original_gate_qubits=qubits, decomposition_part="Exp")
    #         gate_K1l = Gate(tensor=K1l_t, qubits=(q1,), layer_index=layer_index, 
    #                         name="K1l", original_gate_qubits=qubits, decomposition_part="K1l")
    #         gate_K1r = Gate(tensor=K1r_t, qubits=(q2,), layer_index=layer_index, 
    #                         name="K1r", original_gate_qubits=qubits, decomposition_part="K1r")
            
    #                 # Add them to the current GateLayer
    #         gate_layer.gates.extend([gate_K2l, gate_K2r, gate_Exp, gate_K1l, gate_K1r])

    #     circuit.layers.append(gate_layer)

    # print('Weyl decomposition done')
    # circuit.print_gates()
    
    # return circuit

def get_circuit_structure_weyl_layered(n_sites: int, 
                                    raw_gates_per_layer: List[List[jnp.ndarray]], 
                                    layer_is_odd: List[bool], 
                                    hamiltonian: Optional[str] = None) -> Circuit:

    """
    Builds a Circuit object from raw Trotter layers, performing Weyl decomposition 
    and placing components into separate layers.
    K1 -> layer 3*i
    Exp -> layer 3*i + 1
    K2 -> layer 3*i + 2
    """
    circuit = Circuit(n_sites=n_sites, hamiltonian_type=hamiltonian)
    # Use a dictionary to manage the creation of potentially many new layers
    new_layers_map: Dict[int, GateLayer] = {} 

    num_original_layers = len(raw_gates_per_layer)

    for orig_layer_idx, (raw_layer_tensors, orig_is_odd) in enumerate(zip(raw_gates_per_layer, layer_is_odd)):

        # Define the indices for the three new layers derived from this original layer
        k1_layer_idx = 3 * orig_layer_idx
        exp_layer_idx = 3 * orig_layer_idx + 1
        k2_layer_idx = 3 * orig_layer_idx + 2

        # Determine the qubit pairs the original gates acted on
        if orig_is_odd:
            qubit_pairs = [(k, k + 1) for k in range(0, n_sites - 1, 2)] # (0,1), (2,3), ...
        else: # Even layer
            qubit_pairs = [(k, k + 1) for k in range(1, n_sites - 1, 2)] # (1,2), (3,4), ...

        if len(raw_layer_tensors) != len(qubit_pairs):
             print(f"Warning: Original Layer {orig_layer_idx} mismatch. Expected {len(qubit_pairs)} pairs, found {len(raw_layer_tensors)} tensors.")
             # Decide how to handle this - skip layer? raise error? continue cautiously?
             # Let's continue cautiously for now.

        # Ensure the target layers exist in our map
        if k1_layer_idx not in new_layers_map:
            new_layers_map[k1_layer_idx] = GateLayer(layer_index=k1_layer_idx, is_odd=orig_is_odd)
        if exp_layer_idx not in new_layers_map:
            new_layers_map[exp_layer_idx] = GateLayer(layer_index=exp_layer_idx, is_odd=orig_is_odd)
        if k2_layer_idx not in new_layers_map:
            new_layers_map[k2_layer_idx] = GateLayer(layer_index=k2_layer_idx, is_odd=orig_is_odd)

        # Process each 2-qubit gate in the original layer
        for original_tensor, qubits in zip(raw_layer_tensors, qubit_pairs):
            # --- Perform Weyl Decomposition ---
            # Replace this with your actual function call:
            # K1l_t, K1r_t, Exp_t, K2l_t, K2r_t, params_a_b_c = perform_weyl_decomposition(original_tensor) 

            # Placeholder values:
            try:
                # Assume perform_weyl_decomposition exists and returns the tensors + params
                # K1l_t, K1r_t, Exp_t, K2l_t, K2r_t, params_a_b_c = perform_weyl_decomposition(original_tensor)
                
                # Using dummy placeholders for now:
                K1l_t, K1r_t = eye(2), eye(2)
                Exp_t = eye(4) 
                K2l_t, K2r_t = eye(2), eye(2)
                params_a_b_c = (0.0, 0.0, 0.0) # Example parameters for Exp gate

            except Exception as e:
                print(f"Error during Weyl decomposition for gate on qubits {qubits} in original layer {orig_layer_idx}: {e}")
                # Decide how to handle: skip this gate? Use identity? Raise error?
                continue # Skip this problematic gate for now

            q1, q2 = qubits

            # Create Gate objects with CORRECT layer indices
            gate_K1l = Gate(tensor=K1l_t, qubits=(q1,), layer_index=k1_layer_idx, 
                            name="K1l", original_gate_qubits=qubits, decomposition_part="K1l")
            gate_K1r = Gate(tensor=K1r_t, qubits=(q2,), layer_index=k1_layer_idx, 
                            name="K1r", original_gate_qubits=qubits, decomposition_part="K1r")
            
            gate_Exp = Gate(tensor=Exp_t, qubits=qubits, layer_index=exp_layer_idx, 
                            name="ExpXYZ", params=params_a_b_c, original_gate_qubits=qubits, decomposition_part="Exp")
            
            gate_K2l = Gate(tensor=K2l_t, qubits=(q1,), layer_index=k2_layer_idx, 
                            name="K2l", original_gate_qubits=qubits, decomposition_part="K2l")
            gate_K2r = Gate(tensor=K2r_t, qubits=(q2,), layer_index=k2_layer_idx, 
                            name="K2r", original_gate_qubits=qubits, decomposition_part="K2r")
            
            # Add gates to their respective layers using the map
            new_layers_map[k1_layer_idx].gates.extend([gate_K1l, gate_K1r])
            new_layers_map[exp_layer_idx].gates.append(gate_Exp)
            new_layers_map[k2_layer_idx].gates.extend([gate_K2l, gate_K2r])

    # --- Finalize Circuit ---
    # Extract layers from the map and sort them by index
    circuit.layers = sorted(new_layers_map.values(), key=lambda layer: layer.layer_index)

    print(f'Weyl decomposition done. Created {len(circuit.layers)} layers from {num_original_layers} original layers.')
    # circuit.print_gates() # Optional: print summary
    
    return circuit
