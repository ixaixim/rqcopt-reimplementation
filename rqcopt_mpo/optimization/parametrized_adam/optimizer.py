import rqcopt_mpo.jax_config
from typing import Callable, List, Tuple, Dict, Any
import jax.numpy as jnp
import numpy as np          # only for the final “scatter back” to the python side
  
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.utils.pytree import extract_params_tree, update_gate_params_inplace, tree_like_zeros    
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.optimization.parametrized_adam.adam import Adam
from rqcopt_mpo.optimization.parametrized_adam.utils import param_grad_rx, param_grad_ry, param_grad_rz, param_grad_xxyyzz

def optimize(    
    circuit,
    reference_mpo,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    clip_grad_norm: float = None,
    max_steps: int = 1000,
    callback: Callable = None,
    init_vertical_sweep = "bottom-up",
    max_bondim_env : int,
    svd_cutoff: float = 1e-12,
) -> List[float]:

    # TODO: init state m0, v0 to zeros. needs to have the same shape as the param vector 
    params_tree, meta_tree = extract_params_tree(circuit)
    # TODO: construct gate layout with contiguous slices? i.e. 
    opt = Adam(lr, betas, eps, clip_grad_norm,)
    U = opt.prepare_layout_from_trees(params_tree, meta_tree)
    opt.register_param_grad("RX", param_grad_rx)
    opt.register_param_grad("RY", param_grad_ry)
    opt.register_param_grad("RZ", param_grad_rz)
    opt.register_param_grad("Exp(XX+YY+ZZ)", param_grad_xxyyzz)


    def _vert_dir(step: int) -> str:
        """alternate bottom-up / top-down every iteration"""

        if init_vertical_sweep not in ('bottom-up', 'top-down'):
            raise ValueError("vertical_sweep must be one of: 'bottom-up', 'top-down'")
        even_dir  = init_vertical_sweep
        odd_dir   = "bottom-up" if even_dir == "top-down" else "top-down"
        return even_dir if (step % 2 == 0) else odd_dir

    history: List[float] = []

    for it in range(max_steps):
        overlap, grads_ordered, info = cost_and_euclidean_grad(
            circuit,
            reference_mpo,
            vertical_sweep=_vert_dir(it),
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )

    #     # TODO: implement step
        U, state, stats = opt.step(
            U,
            grads_ordered,
            overlap,
            n_sites=circuit.n_sites,
            is_normalized=bool(getattr(reference_mpo, "is_normalized", False)),
        )
        params_tree_updated = opt.unflatten_to_params_tree(U)
        update_gate_params_inplace(circuit, params_tree_updated, meta_tree)

    #     # compute loss
        loss = overlap_to_loss(overlap=overlap, kind='HST', n_sites=circuit.n_sites, normalize=reference_mpo.is_normalized) 
        print(f"Step: {it}, Loss: {loss}")
