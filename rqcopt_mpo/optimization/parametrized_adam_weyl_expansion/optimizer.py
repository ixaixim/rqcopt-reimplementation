import rqcopt_mpo.jax_config

from typing import Callable, List, Tuple

import jax.numpy as jnp

from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.utils.pytree import extract_params_tree
from rqcopt_mpo.optimization.parametrized_adam.adam import Adam
from rqcopt_mpo.optimization.utils import overlap_to_loss

from .utils import (
    parametrize_circuit_weyl,
    compose_weyl_unitary,
    param_grad_weyl15,
)


def _update_circuit_from_trees(circuit, params_tree, meta_tree) -> None:
    """
    In-place update of circuit gates using params_tree/meta_tree.
    Only handles gates with name 'WeylParam2Q' (15 parameters). Other gates
    are left unchanged.
    """
    dtype = getattr(circuit, "dtype", jnp.complex128)
    circuit.sort_layers()
    for layer in circuit.layers:
        L = layer.layer_index
        for gi, g in enumerate(layer.iterate_gates()):
            meta = meta_tree[L][gi]
            theta = params_tree[L][gi]
            if theta.size == 0:
                # non-parametrized gate
                continue
            if meta.get("name") == "WeylParam2Q":
                # single leaf of length 15 expected
                g.params = (theta,)
                g.matrix = compose_weyl_unitary(theta, dtype=dtype)
            else:
                # Leave other gate types untouched in this optimizer variant
                pass


def optimize(
    circuit,
    reference_mpo,
    *,
    lr: float = 1e-3,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    clip_grad_norm: float | None = None,
    bias_correction: bool = True,
    max_steps: int = 1000,
    callback: Callable | None = None,
    init_vertical_sweep: str = "bottom-up",
    max_bondim_env: int,
    svd_cutoff: float = 1e-12,
) -> List[float]:
    """
    Adam on 15-parameter Weyl-expanded 2Q gates while keeping the circuit
    structurally undecomposed.
    """
    # 1) Attach 15-parameter vectors to every 2Q gate
    parametrize_circuit_weyl(circuit)

    # 2) Build params/meta trees and flatten layout
    params_tree, meta_tree = extract_params_tree(circuit)
    opt = Adam(lr, betas, eps, clip_grad_norm, bias_correction)
    U = opt.prepare_layout_from_trees(params_tree, meta_tree)

    # 3) Register param-grad for Weyl gates
    opt.register_param_grad("WeylParam2Q", param_grad_weyl15)

    def _vert_dir(step: int) -> str:
        if init_vertical_sweep not in ("bottom-up", "top-down"):
            raise ValueError("vertical_sweep must be one of: 'bottom-up', 'top-down'")
        even_dir = init_vertical_sweep
        odd_dir = "bottom-up" if even_dir == "top-down" else "top-down"
        return even_dir if (step % 2 == 0) else odd_dir

    history: List[float] = []

    for it in range(max_steps):
        overlap, grads_ordered, _info = cost_and_euclidean_grad(
            circuit,
            reference_mpo,
            vertical_sweep=_vert_dir(it),
            max_bondim_env=max_bondim_env,
            svd_cutoff=svd_cutoff,
        )

        U, _state, _stats = opt.step(
            U,
            grads_ordered,
            overlap,
            n_sites=circuit.n_sites,
            is_normalized=bool(getattr(reference_mpo, "is_normalized", False)),
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

        if callback is not None:
            should_stop = callback(step=it, loss=loss)
            if should_stop:
                return history

    return history
