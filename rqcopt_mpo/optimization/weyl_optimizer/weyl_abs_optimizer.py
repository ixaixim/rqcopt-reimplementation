import rqcopt_mpo.jax_config

from typing import Callable, List, Optional
import jax.numpy as jnp
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.weyl_optimizer.adam import Adam
from rqcopt_mpo.utils.pytree import extract_params_tree
from rqcopt_mpo.optimization.utils import overlap_to_loss

from rqcopt_mpo.optimization.weyl_optimizer.utils import group_and_parametrize_circuit, param_grad_single, param_grad_weyl_abs
from rqcopt_mpo.utils.rotations import _compose_k_from_zyz, _compose_entangler

# optimizer for weyl absorbed gates. 
# assumption: brickwall circuit has been weyl decomposed and absorbed


def _update_circuit_from_trees(circuit, params_tree, meta_tree) -> None:
    """
    Rewrite gate parameters/matrices using the updated parameter tree.
    Supports the absorbed 2-qubit blocks (WeylAbs2Q) and the leading single-qubit layer.
    """
    dtype = getattr(circuit, "dtype", jnp.complex128)
    circuit.sort_layers()

    for layer in circuit.layers:
        layer_idx = layer.layer_index
        for gate_idx, gate in enumerate(layer.iterate_gates()):
            theta = params_tree[layer_idx][gate_idx]
            if theta.size == 0:
                raise ValueError("Theta array is empty: non parametrized gate is not expected for this circuit architecture.")

            meta = meta_tree[layer_idx][gate_idx]
            splits = meta["splits"]

            pieces = []
            start = 0
            for size in splits:
                stop = start + size
                pieces.append(theta[start:stop])
                start = stop

            name = meta["name"]
            if name == "WeylAbs2Q":
                ent_params, left_params, right_params = pieces
                V = _compose_entangler(*ent_params, dtype=dtype)
                Kl = _compose_k_from_zyz(*left_params, dtype=dtype)
                Kr = _compose_k_from_zyz(*right_params, dtype=dtype)
                gate.matrix = jnp.kron(Kl, Kr) @ V
                gate.params = (
                    jnp.asarray(ent_params, dtype=jnp.float64),
                    jnp.asarray(left_params, dtype=jnp.float64),
                    jnp.asarray(right_params, dtype=jnp.float64),
                )
            elif name == "1qLayer0":
                single_params = pieces[0]
                gate.matrix = _compose_k_from_zyz(*single_params, dtype=dtype)
                gate.params = (jnp.asarray(single_params, dtype=jnp.float64),)
            else:
                gate.params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in pieces)

# TODO: check like in RieAdam, whether you should sweep from top or from bottom, from left or from right. It might be necessary to add an optimizer parameter for that.
def optimize(
        circ: Circuit, 
        mpo_ref: MPO, 
        *, 
        max_steps: int,
        max_bondim_env: int,
        svd_cutoff: float = 1e-12,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        clip_grad_norm: float | None = None,
        bias_correction: bool = True,
        callback: Optional[Callable[..., bool]] = None,
):

    # set circuit parameters in place with params and return an array of parameters
    # set the names of the gates to match the registration key (for correct chain rule application)
    new_circ = group_and_parametrize_circuit(circ)
    # get a params tree and an array out of it.
    params_tree, meta_tree = extract_params_tree(new_circ)
    # set up optimizer and divide correclty in slots for proper update in each slot
    opt = Adam(lr, betas, eps, clip_grad_norm, bias_correction)
    params_array = opt.prepare_layout_from_trees(params_tree, meta_tree) # array of parameters

    # register the param keys for the chain rule to be applied correctly on each group of gates
    opt.register_param_grad("WeylAbs2Q", param_grad_weyl_abs)
    opt.register_param_grad("1qLayer0", param_grad_single)



    history: List[float] = []
    # attach the MPO and compute the euclidean gradient
    for it in range(max_steps):
        # euclidean gradient for each gate.
        overlap, grads_ordered, info = cost_and_euclidean_grad(
            new_circ,
            mpo_ref,
            vertical_sweep='bottom-up', # todo: which direction?
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )
        
        params_array, _state, _stats = opt.step(
            params_array,
            grads_ordered,
            overlap,
            n_sites=new_circ.n_sites,
            is_normalized=mpo_ref.is_normalized,
        )

        params_tree_updated = opt.unflatten_to_params_tree(params_array)
        _update_circuit_from_trees(new_circ, params_tree_updated, meta_tree)

        loss = overlap_to_loss(
            overlap=overlap,
            kind="HST",
            n_sites=new_circ.n_sites,
            normalize=mpo_ref.is_normalized,
        )
        history.append(loss)
        print(f"Step: {it}, Loss: {loss}")

        if callback is not None:
            should_stop = callback(
                step=it,
                loss=loss,
            )
            if should_stop:
                break

    return new_circ, history 

    

    

                
