import rqcopt_mpo.jax_config

from typing import Callable, List, Optional
import jax.numpy as jnp
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.weyl_optimizer.adam import Adam
from rqcopt_mpo.utils.pytree import extract_params_tree
from rqcopt_mpo.optimization.utils import overlap_to_loss
from .utils import param_grad_rzz_ising_field, param_grad_rzz_ising_no_field

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
        init_vertical_sweep = "top-down",

        # todo: pass the parametrize function
        # pass the param grad and the names dictionary
):

    # set circuit parameters in place with params and return an array of parameters
    # set the names of the gates to match the registration key (for correct chain rule application)
    # new_circ = parametrize_circuit(circ) # NONE for this circuit. Parameters have already been set
    new_circ = circ.copy()
    # get a params tree and an array out of it.
    params_tree, meta_tree = extract_params_tree(new_circ)
    # set up optimizer and divide correclty in slots for proper update in each slot
    opt = Adam(lr, betas, eps, clip_grad_norm, bias_correction)
    params_array = opt.prepare_layout_from_trees(params_tree, meta_tree) # array of parameters

    # # register the param keys for the chain rule to be applied correctly on each group of gates
    opt.register_param_grad("rzz_ising_with_field", param_grad_rzz_ising_field)
    opt.register_param_grad("rzz_ising_no_field", param_grad_rzz_ising_no_field) # derivative of rzz 



    history: List[float] = []
    # # # attach the MPO and compute the euclidean gradient
    for it in range(max_steps):
        # euclidean gradient for each gate.
        overlap, grads_ordered, info = cost_and_euclidean_grad(
            circ,
            mpo_ref,
            vertical_sweep=init_vertical_sweep, # todo: which direction?
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )
        
    #     # idea: you can use backprop and chain it with the current tensor computation.
        params_array, _state, _stats = opt.step(
            params_array,
            grads_ordered,
            overlap,
            n_sites=new_circ.n_sites,
            is_normalized=mpo_ref.is_normalized,
        )

        params_tree_updated = opt.unflatten_to_params_tree(params_array)
        # _update_circuit_from_trees(new_circ, params_tree_updated, meta_tree)

    #     loss = overlap_to_loss(
    #         overlap=overlap,
    #         kind="HST",
    #         n_sites=new_circ.n_sites,
    #         normalize=mpo_ref.is_normalized,
    #     )
    #     history.append(loss)
    #     print(f"Step: {it}, Loss: {loss}")

    #     if callback is not None:
    #         should_stop = callback(
    #             step=it,
    #             loss=loss,
    #         )
    #         if should_stop:
    #             return new_circ, history

    return new_circ, history 
