import rqcopt_mpo.jax_config

import jax.numpy as jnp
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.parametrized_adam.adam import Adam
from rqcopt_mpo.optimization.weyl_optimizer import utils

# optimizer for weyl absorbed gates. 
# assumption: brickwall circuit has been absorbed using: 
#  

def group_circuit(circ: Circuit) -> Circuit:
    new_layers: list[GateLayer] = []
    # keep the leading single-qubit layer untouched
    new_layers.append(circ.layers[0].copy())


    for pair_idx, two_qubit_layer in enumerate(circ.layers[1::2]):
        single_layer_idx = 2 * pair_idx + 2
        if single_layer_idx >= circ.num_layers:
            raise ValueError("Expecting a single-qubit layer after each two-qubit layer")
        

        single_qubit_layer = circ.layers[single_layer_idx]
        for two_gate in two_qubit_layer.iterate_gates():
            # stash a copy of the 2q matrix so we can absorb its neighbours
            absorbed = two_gate.matrix.copy()

        single_by_qubit = {
            gate.qubits[0]: gate
            for gate in single_qubit_layer.iterate_gates()
        }

        merged_layer = GateLayer(
            layer_index=len(new_layers),
            is_odd=two_qubit_layer.is_odd,
            n_sites=two_qubit_layer.n_sites,
        )

        for two_gate in two_qubit_layer.iterate_gates():
            absorbed = two_gate.matrix.copy()
            q0, q1 = two_gate.qubits
            try:
                left_gate = single_by_qubit[q0]
                right_gate = single_by_qubit[q1]
            except KeyError as err:
                raise ValueError(
                    f"Missing single-qubit dressing for qubits {two_gate.qubits} "
                    f"in layer {single_layer_idx}"
                ) from err

            dressing = jnp.kron(left_gate.matrix, right_gate.matrix)
            absorbed = dressing @ absorbed
            merged_layer.add_gate(
                Gate(
                    matrix=absorbed,
                    qubits=two_gate.qubits,
                    layer_index=merged_layer.layer_index,
                    name=two_gate.name,
                    params=two_gate.params, # probably remove this. 
                    original_gate_qubits=two_gate.original_gate_qubits,
                    decomposition_part=two_gate.decomposition_part,
                )
            )
            
        new_layers.append(merged_layer)

    new_circ = Circuit(    
        n_sites=circ.n_sites,
        dtype=circ.dtype,
        layers=new_layers,
        hamiltonian_type=circ.hamiltonian_type,
        trotter_params=circ.trotter_params, # probably remove this
    )

    return new_circ

def optimize(
        circ: Circuit, 
        mpo_ref: MPO, 
        num_sweeps: int,
        max_bondim_env: int,
        svd_cutoff: float = 1e-12,

):
    # group gates so that contracting the MPO is less expensive
    grouped_circ = group_circuit(circ)

    # set circuit parameters in place with params and return an array of parameters
    # set the names to match the registration key
    params = parametrize_circuit(grouped_circ)

    # set up optimizer (TODO implement adam for array of numbers )
    opt = Adam(lr, betas, eps, clip_grad_norm, bias_correction)
    U = opt.prepare_layout_from_trees(params_tree, meta_tree)

    opt.register_param_grad("WeylAbs2Q", param_grad_weyl15)
    opt.register_param_grad("1qLayer0", param_grad_weyl15)


    history: List[float] = []
    # attach the MPO and compute the euclidean gradient
    for it in range(num_sweeps):
        # euclidean gradient for each gate.
        overlap, grads_ordered, info = cost_and_euclidean_grad(
            grouped_circ,
            mpo_ref,
            vertical_sweep='bottom-up', # todo: which direction?
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )
        
        params, _state, _stats = opt.step(
            params,
            grads_ordered,
            overlap,
            n_sites=grouped_circ.n_sites,
            is_normalized=mpo_ref.is_normalized,
        )

        params_tree_updated = opt.unflatten_to_params_tree(U)
        _update_circuit_from_trees(circuit, params_tree_updated, meta_tree)

        loss = overlap_to_loss(
            overlap=overlap,
            kind="HST",
            n_sites=circuit.n_sites,
            normalize=getattr(reference_mpo, "is_normalized", False),
        )
        history.append(loss)
        print(f"Step: {it}, Loss: {loss}")

    return history 

    

    

                

