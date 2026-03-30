import numpy as np
from qiskit.synthesis import TwoQubitWeylDecomposition
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit

def analyze_circuit_weyl_redundancy(circuit: Circuit, threshold: float = 1e-5):
    """
    Analyzes the Weyl decomposition of each 2-qubit gate in the circuit's layers.
    For each 2-qubit layer, it determines how many Weyl parameters (a, b, c) are 
    identically zero across ALL gates in that layer.
    
    A standard 2-qubit gate requires 3 physical CNOT layers in its synthesis.
    If 'n' Weyl parameters are zero for all gates in a layer, that layer only 
    requires (3 - n) physical layers.
    """
    print(f"\n--- Weyl Redundancy Analysis for Circuit ({circuit.n_sites} sites) ---")
    total_physical_layers = 0
    
    for layer in circuit.layers:
        two_q_gates = [g for g in layer.gates if g.is_two_qubit()]
        if not two_q_gates:
            # Single-qubit layers or empty layers don't contribute to 2-qubit depth
            continue
        
        layer_a = []
        layer_b = []
        layer_c = []
        
        for gate in two_q_gates:
            # decomp.a, decomp.b, decomp.c are the Weyl coordinates
            # Qiskit convention: pi/4 >= a >= b >= |c|
            decomp = TwoQubitWeylDecomposition(gate.matrix)
            layer_a.append(decomp.a)
            layer_b.append(decomp.b)
            layer_c.append(decomp.c)
            
        # Check if all gates in this layer have zero parameters
        all_a_zero = all(abs(a) < threshold for a in layer_a)
        all_b_zero = all(abs(b) < threshold for b in layer_b)
        all_c_zero = all(abs(c) < threshold for c in layer_c)
        
        n_zero = 0
        if all_c_zero:
            n_zero += 1
            if all_b_zero:
                n_zero += 1
                if all_a_zero:
                    n_zero += 1
        
        physical_layers = 3 - n_zero
        total_physical_layers += physical_layers
        
        print(f"Layer {layer.layer_index:2d}: {len(two_q_gates):2d} gates | "
              f"Zero parameters: {n_zero} | Physical cost: {physical_layers}")
              
    print(f"Total estimated physical 2-qubit layers: {total_physical_layers}")
    print("-----------------------------------------------------------\n")
    
    return total_physical_layers
