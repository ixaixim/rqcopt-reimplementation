import jax
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, GateLayer, Gate
from rqcopt_mpo.mpo.mpo_dataclass import MPO

def register_pytrees():
    # Gate
    def flatten_gate(gate):
        children = (gate.matrix, gate.params, gate.params_dict)
        aux_data = {
            'qubits': gate.qubits,
            'layer_index': gate.layer_index,
            'name': gate.name,
            'original_gate_qubits': gate.original_gate_qubits,
            'decomposition_part': gate.decomposition_part
        }
        return children, aux_data

    def unflatten_gate(aux_data, children):
        matrix, params, params_dict = children
        return Gate(
            matrix=matrix,
            qubits=aux_data['qubits'],
            layer_index=aux_data['layer_index'],
            name=aux_data['name'],
            params=params,
            params_dict=params_dict,
            original_gate_qubits=aux_data['original_gate_qubits'],
            decomposition_part=aux_data['decomposition_part']
        )
    
    jax.tree_util.register_pytree_node(Gate, flatten_gate, unflatten_gate)

    # GateLayer
    def flatten_layer(layer):
        children = (layer.gates,)
        aux_data = {'layer_index': layer.layer_index, 'is_odd': layer.is_odd, 'n_sites': layer.n_sites}
        return children, aux_data

    def unflatten_layer(aux_data, children):
        return GateLayer(
            layer_index=aux_data['layer_index'],
            is_odd=aux_data['is_odd'],
            gates=children[0],
            n_sites=aux_data['n_sites']
        )

    jax.tree_util.register_pytree_node(GateLayer, flatten_layer, unflatten_layer)

    # Circuit
    def flatten_circuit(circuit):
        children = (circuit.layers,)
        aux_data = {
            'n_sites': circuit.n_sites,
            'dtype': circuit.dtype,
            'hamiltonian_type': circuit.hamiltonian_type,
            'trotter_params': circuit.trotter_params
        }
        return children, aux_data

    def unflatten_circuit(aux_data, children):
        return Circuit(
            n_sites=aux_data['n_sites'],
            dtype=aux_data['dtype'],
            layers=children[0],
            hamiltonian_type=aux_data['hamiltonian_type'],
            trotter_params=aux_data['trotter_params']
        )

    jax.tree_util.register_pytree_node(Circuit, flatten_circuit, unflatten_circuit)
    
    # MPO
    def flatten_mpo(mpo):
        children = (mpo.tensors,)
        aux_data = {
            'is_left_canonical': mpo.is_left_canonical,
            'is_right_canonical': mpo.is_right_canonical,
            'norm': mpo.norm,
            'is_normalized': mpo.is_normalized
        }
        return children, aux_data
        
    def unflatten_mpo(aux_data, children):
        return MPO(
            tensors=children[0],
            is_left_canonical=aux_data['is_left_canonical'],
            is_right_canonical=aux_data['is_right_canonical'],
            norm=aux_data['norm'],
            is_normalized=aux_data['is_normalized']
        )
        
    jax.tree_util.register_pytree_node(MPO, flatten_mpo, unflatten_mpo)

# Register on import
register_pytrees()
