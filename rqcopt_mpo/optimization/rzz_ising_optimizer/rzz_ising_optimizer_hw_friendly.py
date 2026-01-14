import rqcopt_mpo.jax_config
import jax.numpy as jnp
from typing import Callable, List, Optional

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.optimization.gradient import cost_and_euclidean_grad
from rqcopt_mpo.optimization.weyl_optimizer.adam import Adam
from rqcopt_mpo.utils.pytree import extract_params_tree
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.optimization.rzz_ising_optimizer.utils import (
    _rzz_matrix, _drzz_matrix, _compose_k_from_zyz, 
    _paulis_1q, _rot_from_generator, _d_rot_from_generator,
    _backprop_hst_loss
)

def _dK_parts(p, dtype):
    """
    Computes K and its derivatives w.r.t parameters [th, psi, ph].
    K = Rz(th) Ry(psi) Rz(ph).
    Returns ([dK_dth, dK_dpsi, dK_dph], K_matrix).
    """
    th, psi, ph = p[0], p[1], p[2]
    _, Y, Z, _ = _paulis_1q(dtype)
    
    Rz_th = _rot_from_generator(th, Z, 0.5, dtype)
    Ry_psi = _rot_from_generator(psi, Y, 0.5, dtype)
    Rz_ph = _rot_from_generator(ph, Z, 0.5, dtype)
    
    dRz_th = _d_rot_from_generator(th, Z, 0.5, dtype)
    dRy_psi = _d_rot_from_generator(psi, Y, 0.5, dtype)
    dRz_ph = _d_rot_from_generator(ph, Z, 0.5, dtype)
    
    K = Rz_th @ Ry_psi @ Rz_ph
    
    d_th = dRz_th @ Ry_psi @ Rz_ph
    d_psi = Rz_th @ dRy_psi @ Rz_ph
    d_ph = Rz_th @ Ry_psi @ dRz_ph
    
    return [d_th, d_psi, d_ph], K

def param_grad_rzz_ising_1q(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Gradient for Ising_hw_1q gate.
    Params: 3 (phi, theta, lam).
    U = Rz(phi) Ry(theta) Rz(lam).
    """
    dtype = dL_dG.dtype
    
    # theta is [phi, theta, lam] which matches our _dK_parts convention
    derivs_K, K = _dK_parts(theta, dtype)
    
    grads = [
        _backprop_hst_loss(L, dL_dG, dU, n_sites, is_normalized)
        for dU in derivs_K
    ]
    return jnp.stack(grads)

def param_grad_rzz_ising_1rzz(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Gradient for Ising_hw_1rzz gate.
    Params: 1 (angle).
    U = RZZ(angle).
    """
    dtype = dL_dG.dtype
    
    p_rzz = theta[0]
    
    # 1. Build Matrices and Derivatives
    dRZZ = _drzz_matrix(p_rzz, dtype)
    
    derivs = [dRZZ]
        
    # Stack and contract
    grads = [
        _backprop_hst_loss(L, dL_dG, dU, n_sites, is_normalized)
        for dU in derivs
    ]
    return jnp.stack(grads)

def _update_circuit_from_trees_hw(circuit, params_tree, meta_tree):
    dtype = getattr(circuit, "dtype", jnp.complex128)
    circuit.sort_layers()
    kron = jnp.kron
    
    for layer in circuit.layers:
        layer_idx = layer.layer_index
        if layer_idx not in params_tree:
            continue
            
        for gate_idx, gate in enumerate(layer.iterate_gates()):
            if gate_idx >= len(params_tree[layer_idx]):
                continue
                
            theta = params_tree[layer_idx][gate_idx]
            if theta.size == 0:
                continue

            meta = meta_tree[layer_idx][gate_idx]
            name = meta["name"]
            
            if name == "Ising_hw_1rzz":
                # Reconstruct
                p_rzz = theta[0]
                
                RZZ = _rzz_matrix(p_rzz, dtype=dtype)
                
                gate.matrix = RZZ
                
                # Update params in gate (splitting array back to tuple logic)
                # Helper to split based on original shapes
                splits = meta["splits"]
                pieces = []
                start = 0
                for size in splits:
                    stop = start + size
                    pieces.append(theta[start:stop])
                    start = stop
                gate.params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in pieces)
                
            elif name == "Ising_hw_1q":
                p = theta
                K = _compose_k_from_zyz(p[0], p[1], p[2], dtype=dtype)
                gate.matrix = K
                gate.params = (jnp.asarray(p, dtype=jnp.float64),)

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
):
    new_circ = circ.copy()
    params_tree, meta_tree = extract_params_tree(new_circ)
    
    opt = Adam(lr, betas, eps, clip_grad_norm, bias_correction)
    params_array = opt.prepare_layout_from_trees(params_tree, meta_tree)

    # Register our HW friendly gradient
    opt.register_param_grad("Ising_hw_1rzz", param_grad_rzz_ising_1rzz)
    opt.register_param_grad("Ising_hw_1q", param_grad_rzz_ising_1q)

    history: List[float] = []
    
    for it in range(max_steps):
        overlap, grads_ordered, info = cost_and_euclidean_grad(
            new_circ,
            mpo_ref,
            vertical_sweep=init_vertical_sweep, 
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
        _update_circuit_from_trees_hw(new_circ, params_tree_updated, meta_tree)

        loss = overlap_to_loss(
            overlap=overlap,
            kind="HST",
            n_sites=new_circ.n_sites,
            normalize=mpo_ref.is_normalized,
        )
        history.append(loss)
        print(f"Step: {it}, Loss: {loss}")

        if callback is not None:
            should_stop = callback(step=it, loss=loss)
            if should_stop:
                return new_circ, history

    return new_circ, history
