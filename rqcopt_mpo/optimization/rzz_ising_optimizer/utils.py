import rqcopt_mpo.jax_config
import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.utils.rotations import (
    _backprop_hst_loss,
    _compose_k_from_zyz,
    _paulis_1q,
    _rot_from_generator,
    _d_rot_from_generator
)

def _rzz_matrix(theta: float, dtype=jnp.complex128):
    """
    RZZ(theta) = exp(-i * theta/2 * Z⊗Z)
    Diagonal elements: exp(-it/2), exp(it/2), exp(it/2), exp(-it/2)
    """
    # theta is a scalar array or float
    phase_m = jnp.exp(-1j * theta / 2)
    phase_p = jnp.exp(1j * theta / 2)
    # diag: [m, p, p, m]
    return jnp.diag(jnp.array([phase_m, phase_p, phase_p, phase_m], dtype=dtype))

def _drzz_matrix(theta: float, dtype=jnp.complex128):
    """
    Derivative of RZZ(theta) w.r.t theta.
    d/dtheta exp(-i theta/2 * ZZ) = -i/2 * ZZ * exp(...)
    Since RZZ is diagonal, this is element-wise mult by diag(-i/2, i/2, i/2, -i/2) 
    So we multilpy the diagonal elements by [-i/2, i/2, i/2, -i/2].
    """
    rzz = _rzz_matrix(theta, dtype)
    factors = jnp.array([-0.5j, 0.5j, 0.5j, -0.5j], dtype=dtype)
    return jnp.diag(factors) @ rzz


def _update_circuit_from_trees(circuit, params_tree, meta_tree) -> None:
    """
    Rewrite gate parameters/matrices using the updated parameter tree.
    Supports "Ising_field_term" and "Ising_no_field".
    """
    dtype = getattr(circuit, "dtype", jnp.complex128)
    circuit.sort_layers()
    kron = jnp.kron

    for layer in circuit.layers:
        layer_idx = layer.layer_index
        for gate_idx, gate in enumerate(layer.iterate_gates()):
            theta = params_tree[layer_idx][gate_idx]
            if theta.size == 0:
                continue

            meta = meta_tree[layer_idx][gate_idx]
            splits = meta["splits"]

            pieces = []
            start = 0
            for size in splits:
                stop = start + size
                pieces.append(theta[start:stop])
                start = stop

            name = meta["name"]
            
            if name == "Ising_no_field":
                # Single RZZ angle
                if len(pieces) != 1:
                     # It might be that theta is just a single array of size 1?
                     # splits would be [1]
                     pass
                
                angle = jnp.asarray(pieces[0], dtype=jnp.float64).item()
                gate.matrix = _rzz_matrix(angle, dtype=dtype)
                gate.params = (jnp.asarray(pieces[0], dtype=jnp.float64),)

            elif name == "Ising_field_term":
                # Expected splits: 8 parts
                # K_L_upper(3), K_L_lower(3), Rzz1(1), mid_upper(3), mid_lower(3), Rzz2(1), K_R_upper(3), K_R_lower(3)
                if len(pieces) != 8:
                    raise ValueError(f"Ising_field_term gate expects 8 parameter blocks, got {len(pieces)}")
                
                # Unpack params
                k_l_u = pieces[0]
                k_l_l = pieces[1]
                rzz1_ang = pieces[2].item()
                mid_u = pieces[3]
                mid_l = pieces[4]
                rzz2_ang = pieces[5].item()
                k_r_u = pieces[6]
                k_r_l = pieces[7]

                # Reconstruct Matrices: U3(th, ph, lam) -> Rz(ph) Ry(th) Rz(lam)
                # _compose_k_from_zyz(rz1, ry, rz2)
                # params are [th, ph, lam]
                Ku_L = _compose_k_from_zyz(k_l_u[1], k_l_u[0], k_l_u[2], dtype=dtype)
                Kl_L = _compose_k_from_zyz(k_l_l[1], k_l_l[0], k_l_l[2], dtype=dtype)
                K_L = kron(Ku_L, Kl_L)

                RZZ1 = _rzz_matrix(rzz1_ang, dtype=dtype)

                Ku_mid = _compose_k_from_zyz(mid_u[1], mid_u[0], mid_u[2], dtype=dtype)
                Kl_mid = _compose_k_from_zyz(mid_l[1], mid_l[0], mid_l[2], dtype=dtype)
                K_mid = kron(Ku_mid, Kl_mid)

                RZZ2 = _rzz_matrix(rzz2_ang, dtype=dtype)

                Ku_R = _compose_k_from_zyz(k_r_u[1], k_r_u[0], k_r_u[2], dtype=dtype)
                Kl_R = _compose_k_from_zyz(k_r_l[1], k_r_l[0], k_r_l[2], dtype=dtype)
                K_R = kron(Ku_R, Kl_R)

                # Combine: K_L @ RZZ1 @ K_mid @ RZZ2 @ K_R
                # Note: Matrix multiplication order. 
                # If circuit is K_L then RZZ1... it means K_L acts first? No, usually last applied is left.
                # In Qiskit/Circuit builder: `qc.append(K_L)`. 
                # If state is |psi>, after K_L it is K_L |psi>.
                # Then RZZ1. RZZ1 K_L |psi>.
                # So matrix is ... @ RZZ1 @ K_L.
                # BUT `gate.matrix` usually stores the full operation such that `Mat @ vec` is the result.
                # The circuit builder appends K2l, K2r (Left), then RZZ, then Middle, then RZZ, then K1 (Right).
                # Operations appended later act later.
                # So U = K_R @ RZZ2 @ K_mid @ RZZ1 @ K_L
                # Wait, "K_L" usually refers to the left-most in the diagram (first applied).
                # "Left" in tensor network / brickwall usually means "first layer" or "left side of the tensor".
                # But if `K2l` is appended FIRST in the quantum circuit, it is applied FIRST to the state.
                # So U = ... @ Second @ First.
                # So if K_L is appended first, it is right-most in matrix multiplication.
                # U = K_R @ RZZ2 @ K_mid @ RZZ1 @ K_L.
                
                gate.matrix = K_R @ RZZ2 @ K_mid @ RZZ1 @ K_L
                
                # Update params
                gate.params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in pieces)
            
            else:
                 # Fallback or error
                 gate.params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in pieces)


def param_grad_rzz_ising_no_field(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Chain rule for Ising_no_field (just RZZ).
    theta is [angle].
    """
    dtype = dL_dG.dtype
    angle = theta[0]
    
    dU_dtheta = _drzz_matrix(angle, dtype=dtype)
    
    grad = _backprop_hst_loss(L, dL_dG, dU_dtheta, n_sites, is_normalized)
    return jnp.array([grad])


def param_grad_rzz_ising_field(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Chain rule for Ising_field_term.
    Params: 20 params flattened.
    Structure: K_L @ RZZ1 @ K_mid @ RZZ2 @ K_R  (Sequence in circuit: K_L, RZZ1, K_mid, RZZ2, K_R)
    Matrix U = K_R @ RZZ2 @ K_mid @ RZZ1 @ K_L
    """
    dtype = dL_dG.dtype
    kron = jnp.kron
    
    # Unpack (slicing)
    # theta is 1D array
    # 0-3: K_L_upper
    # 3-6: K_L_lower
    # 6: Rzz1
    # 7-10: mid_upper
    # 10-13: mid_lower
    # 13: Rzz2
    # 14-17: K_R_upper
    # 17-20: K_R_lower
    
    p_klu = theta[0:3]
    p_kll = theta[3:6]
    p_rzz1 = theta[6]
    p_midu = theta[7:10]
    p_midl = theta[10:13]
    p_rzz2 = theta[13]
    p_kru = theta[14:17]
    p_krl = theta[17:20]
    
    # Helpers
    def make_K(p): return _compose_k_from_zyz(p[1], p[0], p[2], dtype=dtype)
    # _d_rot_from_generator: (angle, G, 0.5, dtype)
    _, Y, Z, I2 = _paulis_1q(dtype)
    
    def dK_parts(p):
        # p = [theta, phi, lam]
        # K = Rz(phi) Ry(theta) Rz(lam)
        th, ph, lam = p[0], p[1], p[2]
        
        Rz_ph = _rot_from_generator(ph, Z, 0.5, dtype)
        Ry_th = _rot_from_generator(th, Y, 0.5, dtype)
        Rz_lam = _rot_from_generator(lam, Z, 0.5, dtype)
        
        dRz_ph = _d_rot_from_generator(ph, Z, 0.5, dtype)
        dRy_th = _d_rot_from_generator(th, Y, 0.5, dtype)
        dRz_lam = _d_rot_from_generator(lam, Z, 0.5, dtype)
        
        # dK/dth (p[0])
        dth = Rz_ph @ dRy_th @ Rz_lam
        
        # dK/dph (p[1])
        dph = dRz_ph @ Ry_th @ Rz_lam
        
        # dK/dlam (p[2])
        dlam = Rz_ph @ Ry_th @ dRz_lam
        
        return [dth, dph, dlam], (Rz_ph @ Ry_th @ Rz_lam)

    # Build components and derivatives
    dKlu_list, Klu = dK_parts(p_klu)
    dKll_list, Kll = dK_parts(p_kll)
    KL = kron(Klu, Kll)
    
    RZZ1 = _rzz_matrix(p_rzz1, dtype=dtype)
    dRZZ1 = _drzz_matrix(p_rzz1, dtype=dtype)
    
    dMidu_list, Midu = dK_parts(p_midu)
    dMidl_list, Midl = dK_parts(p_midl)
    KMid = kron(Midu, Midl)
    
    RZZ2 = _rzz_matrix(p_rzz2, dtype=dtype)
    dRZZ2 = _drzz_matrix(p_rzz2, dtype=dtype)
    
    dKru_list, Kru = dK_parts(p_kru)
    dKrl_list, Krl = dK_parts(p_krl)
    KR = kron(Kru, Krl)
    
    # U = KR @ RZZ2 @ KMid @ RZZ1 @ KL
    # Let's assemble dU/d_param
    
    derivs = []
    
    # 1. K_L params (first applied, so right-most)
    # dU/d_param = KR @ RZZ2 @ KMid @ RZZ1 @ dKL
    
    remainder_L = KR @ RZZ2 @ KMid @ RZZ1
    
    # K_L_upper (phi, th, lam)
    for dK in dKlu_list:
        dKL = kron(dK, Kll)
        derivs.append(remainder_L @ dKL)
        
    # K_L_lower
    for dK in dKll_list:
        dKL = kron(Klu, dK)
        derivs.append(remainder_L @ dKL)
        
    # 2. RZZ1
    # dU/d_rzz1 = KR @ RZZ2 @ KMid @ dRZZ1 @ KL
    derivs.append(KR @ RZZ2 @ KMid @ dRZZ1 @ KL)
    
    # 3. Middle
    # dU/d_mid = KR @ RZZ2 @ dKMid @ RZZ1 @ KL
    remainder_mid_left = KR @ RZZ2
    remainder_mid_right = RZZ1 @ KL
    
    # mid upper
    for dK in dMidu_list:
        dKMid = kron(dK, Midl)
        derivs.append(remainder_mid_left @ dKMid @ remainder_mid_right)
        
    # mid lower
    for dK in dMidl_list:
        dKMid = kron(Midu, dK)
        derivs.append(remainder_mid_left @ dKMid @ remainder_mid_right)
        
    # 4. RZZ2
    # dU/d_rzz2 = KR @ dRZZ2 @ KMid @ RZZ1 @ KL
    derivs.append(KR @ dRZZ2 @ KMid @ RZZ1 @ KL)
    
    # 5. K_R (last applied, left-most)
    # dU/d_KR = dKR @ RZZ2 @ KMid @ RZZ1 @ KL
    remainder_R = RZZ2 @ KMid @ RZZ1 @ KL
    
    # K_R_upper
    for dK in dKru_list:
        dKR = kron(dK, Krl)
        derivs.append(dKR @ remainder_R)
        
    # K_R_lower
    for dK in dKrl_list:
        dKR = kron(Kru, dK)
        derivs.append(dKR @ remainder_R)
        
    # Stack grads
    grads = [
        _backprop_hst_loss(L, dL_dG, dU, n_sites, is_normalized)
        for dU in derivs
    ]
    return jnp.stack(grads)