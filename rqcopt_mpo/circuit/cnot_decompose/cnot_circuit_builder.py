import rqcopt_mpo.jax_config

from rqcopt_mpo.circuit.circuit_dataclasses import GateLayer, Circuit
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.quantum_info import Operator
from qiskit.circuit.library import CXGate

# decomposes the given circuit into cnot gates, given a fixed fidelity threshold
# absorbs the single qubit layer at one end (depending on how many cnots are present in the gate). 
#   reduces degrees of freedom.
basis = CXGate()
decomposer = TwoQubitBasisDecomposer(CXGate())
def cnot_decompose_circuit(orig: Circuit, basis_fidelity: None) -> Circuit:
    """
    Replace a two-qubit Gate into CNOT and arbitrary single qubit gates.
    Absorbs the 1q gates of adjacent layers together to reduce free parameters. If 1q gate lies on the boundary, absorb with 1q gate of the next even layer.
    Assumes brickwall layer, even num of qubits, and start with even layer (0-1, 2-3, (n-2)-(n-1)). 
    """

    # we are mapping from a circuit with n layers to a circuit with n+1 layers,
    #   the first layer is 1q gates, the other layers are parametrized 2q gates.
    # new layer 
    new_layers: dict[int, GateLayer] = {} 
    layer = GateLayer()
    prev_gate_list = # is a list of jnp.identity(2) 1q matrices 
    idx = 0
    for lay in orig.layers:
        for g in lay.iterate_gates():
            # decompose gate into cnot. 
            circ = decomposer(Operator(g.matrix))
            for instr in circ.data:
                op = instr.operation # use to_matrix() to have the matrix
                # count how many cnots.
                cnot_num = 0
                if op.num_qubits == 2: 
                    cnot_num += 1
                
            # divide params within gate in three categories gr1, gr2 gr3, divide into the first and last single qubit gates before the first cnot and after the last cnot and all the gates in between the cnots (if only one cnot is present, this category contains only the single cnot).
            gr1, gr2, gr3 = 
            
            # the very first layer is 1q gates
            if idx == 0: 
                layer.gates.append(gr1, idx = idx)
                # save the other params for the next layer
                idx +=1
                layer.gates.append(matrix = gr2@gr3, params = params_gr2-gr3)
                boundary_qubit_matrix = gr3
                # increement lay
                continue 
            else:
                # absorption: the previous idx gates absorb the new gr1 single q gates.
                if qubits not 0 or n-1:
                    previous_matrix = layer[idx-1].gate.matrix
                    layer[idx-1].gate.matrix = gr1 @ previous_matrix
                    layer[idx-1].gate.params = layer[idx-1].gate.params.append(params_gr1)

                if layer is even: 
                    # absorb the boundary qubits:
                    previous_matrix = layer[idx-2].gate.matrix
                    layer[idx-2].gate.matrix = gr1 @ previous_matrix 
                    layer[idx-2].gate.params = layer[idx-2].gate.params.append(params_gr1)

                # save current matrix and params gate
                layer.gates.append(matrix = gr2@gr3, params = params_gr2-gr3)
                idx+=1
                
    return Circuit()

        
                

                

         



