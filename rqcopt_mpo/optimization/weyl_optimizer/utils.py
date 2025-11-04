from rqcopt_mpo.optimization... import # _backprop_hst_loss, _rot_from_generator, and all other necessary functions from the other utils.py module

def parametrize_circuit(circuit) -> None: # returns a pytree
    """Mutate the circuit in place. Appends parameters to each gate.
    each 2q gate consists of an entangler SU(4) (3 angles), and the accompanying upper/lower 1q gates to its left (3 angles each). 
    total: 9
    the 1q layer at the beginning consists of 1q layers only (3 angles each). 
    """
    for lay_idx, lay in enumerate(circuit.layers):
        for g in lay.gates:
            if l == 0: 
                # 1q gates only 
                # do rzrxrz decomposition 
                # set params of g
                # set name to "1qLayer0"
            else:
                # 2q gates only
                # set name to "WeylAbs2Q"
                # set params of g
                
                
def param_grad_weyl_abs(
        theta: 
        dL_dG:
        L:
        meta:
        n_sites:
        is_normalized:
) -> jnp.ndarray:
    """
    Map Euclidean gradient dL/dG for a 2-qubit gate to dL/dtheta for the 9D
    Weyl parameterization U = V (Kl ⊗ Kr).

    Parameter ordering: (,b,c, t1,p1,f1, t2,p2,f2)
    with K = Rz(t) @ Ry(p) @ Rz(f), scales 0.5 for 1q, 1.0 for entangler.
    """
    dtype = dL_dG.dtype
    X, Y, Z, _ = _paulis_1q(dtype)
    XX, YY, ZZ, _I4 = _paulis_2q(dtype)
    kron = jnp.kron

    a, b, c, t1, p1, f1, t2, p2, f2 = theta

    # Build individual Rz/Ry for reuse (to form K and their derivatives)
    V = _compose_entangler(a, b, c, dtype=dtype)

    Rz1_3 = _rot_from_generator(t3, Z, 0.5, dtype)
    Ry3   = _rot_from_generator(p3, Y, 0.5, dtype)
    Rz2_3 = _rot_from_generator(f3, Z, 0.5, dtype)
    Kl   = Rz1_3 @ Ry3 @ Rz2_3

    Rz1_4 = _rot_from_generator(t4, Z, 0.5, dtype)
    Ry4   = _rot_from_generator(p4, Y, 0.5, dtype)
    Rz2_4 = _rot_from_generator(f4, Z, 0.5, dtype)
    Kr   = Rz1_4 @ Ry4 @ Rz2_4

    B = kron(K1l, K1r)

    # Entangler derivatives (commuting)
    s2q = 1.0
    dV_da = (1j * s2q) * (XX @ V)
    dV_db = (1j * s2q) * (YY @ V)
    dV_dc = (1j * s2q) * (ZZ @ V)

    # Derivatives for right-side 1q factors
    s1q = 0.5
    dK1l_dt = (-1j * s1q) * (Z @ K1l)
    dK1l_dp = Rz1_3 @ ((-1j * s1q) * (Y @ Ry3)) @ Rz2_3
    dK1l_df = Rz1_3 @ Ry3 @ ((-1j * s1q) * (Z @ Rz2_3))

    dK1r_dt = (-1j * s1q) * (Z @ K1r)
    dK1r_dp = Rz1_4 @ ((-1j * s1q) * (Y @ Ry4)) @ Rz2_4
    dK1r_df = Rz1_4 @ Ry4 @ ((-1j * s1q) * (Z @ Rz2_4))

    # Assemble dU/dθ blocks
    derivs: list[jnp.ndarray] = []



    # Entangler parameters
    derivs.append(dV_da @ B)
    derivs.append(dV_db @ B)
    derivs.append(dV_dc @ B)

    # Right-local parameters (K1l ⊗ K1r)
    derivs.append(V @ kron(dK1l_dt, K1r))
    derivs.append(V @ kron(dK1l_dp, K1r))
    derivs.append(V @ kron(dK1l_df, K1r))

    derivs.append(V @ kron(K1l, dK1r_dt))
    derivs.append(V @ kron(K1l, dK1r_dp))
    derivs.append(V @ kron(K1l, dK1r_df))

    # Backprop each derivative into a scalar gradient
    grads = [
        _backprop_hst_loss(L, dL_dG, dU_dth, n_sites, is_normalized)
        for dU_dth in derivs
    ]
    return jnp.stack(grads)