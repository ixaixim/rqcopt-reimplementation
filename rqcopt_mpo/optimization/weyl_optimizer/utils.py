import rqcopt_mpo.jax_config
# from rqcopt_mpo.optimization... import # _backprop_hst_loss, _rot_from_generator, and all other necessary functions from the other utils.py module
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
import jax.numpy as jnp
from qiskit.synthesis import OneQubitEulerDecomposer
from rqcopt_mpo.utils.rotations import _rot_from_generator, _paulis_1q, _paulis_2q, _backprop_hst_loss
from rqcopt_mpo.utils.rotations import _compose_k_from_zyz, _compose_entangler



def group_and_parametrize_circuit(circ: Circuit) -> Circuit:
    """
    Parametrizes an absorbed weyl circuit and Groups the gate together for faster MPO sweeping.
    Mutate the circuit in place. Appends parameters to each gate.
    each 2q gate consists of an entangler SU(4) (3 angles), and the accompanying upper/lower 1q gates to its left (3 angles each). 
    total: 9
    the 1q layer at the beginning consists of 1q layers only (3 angles each). 
    """
    new_layers: list[GateLayer] = []
    decomp = OneQubitEulerDecomposer(basis="ZYZ")

    # LAYER 0
    # keep the leading single-qubit layer untouched and add Euler parameters to it
    for single_qubit_gate in circ.layers[0].iterate_gates():
        th, ph, lam, _ = decomp.angles_and_phase(single_qubit_gate.matrix)
        single_params = jnp.array([ph, th, lam], dtype=jnp.float64)
        single_qubit_gate.params = (single_params,)
        single_qubit_gate.name = "1qLayer0"
    new_layers.append(circ.layers[0].copy())

    # ALL OTHER LAYERS
    for pair_idx, two_qubit_layer in enumerate(circ.layers[1::2]):
        single_layer_idx = 2 * pair_idx + 2
        if single_layer_idx >= circ.num_layers:
            raise ValueError("Expecting a single-qubit layer after each two-qubit layer")
        

        single_qubit_layer = circ.layers[single_layer_idx]
        single_by_qubit: dict[int, Gate] = {}
        single_by_qubit_params: dict[int, jnp.ndarray] = {}
        for gate in single_qubit_layer.iterate_gates():
            th, ph, lam, _ = decomp.angles_and_phase(gate.matrix)
            params_vec = jnp.array([ph, th, lam], dtype=jnp.float64) # pay attention to the order.
            single_by_qubit[gate.qubits[0]] = gate
            single_by_qubit_params[gate.qubits[0]] = params_vec
            gate.params = (params_vec,)

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
            entangler_params = jnp.array(two_gate.params, dtype=jnp.float64)
            
            left_params = single_by_qubit_params[q0]
            right_params = single_by_qubit_params[q1]
            merged_params = (
                entangler_params,
                left_params,
                right_params,
            )
            merged_layer.add_gate(
                Gate(
                    matrix=absorbed,
                    qubits=two_gate.qubits,
                    name="WeylAbs2Q",
                    layer_index=merged_layer.layer_index,
                    params=merged_params,
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


                
                
def param_grad_weyl_abs(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Map Euclidean gradient dL/dG for a 2-qubit gate to dL/dtheta for the 9D
    Weyl parameterization U = V (Kl ⊗ Kr).
    where V is the SU(4) entangler

    Parameter ordering: (a,b,c, t1,p1,f1, t2,p2,f2)
    with K = Rz(t) @ Ry(p) @ Rz(f), scales 0.5 for 1q.
    """
    dtype = dL_dG.dtype
    X, Y, Z, _ = _paulis_1q(dtype)
    XX, YY, ZZ, _I4 = _paulis_2q(dtype)
    kron = jnp.kron

    a, b, c, t1, p1, f1, t2, p2, f2 = theta

    # Build individual Rz/Ry for reuse (to form K and their derivatives)
    V = _compose_entangler(a, b, c, dtype=dtype)

    Rz_1l = _rot_from_generator(t1, Z, 0.5, dtype)
    Ryl   = _rot_from_generator(p1, Y, 0.5, dtype)
    Rz_2l = _rot_from_generator(f1, Z, 0.5, dtype)
    Kl   = Rz_1l @ Ryl @ Rz_2l

    Rz_1r = _rot_from_generator(t2, Z, 0.5, dtype)
    Ryr   = _rot_from_generator(p2, Y, 0.5, dtype)
    Rz_2r = _rot_from_generator(f2, Z, 0.5, dtype)
    Kr   = Rz_1r @ Ryr @ Rz_2r

    B = kron(Kl, Kr)

    # Entangler derivatives (commuting)
    dV_da = 1j * (XX @ V)
    dV_db = 1j * (YY @ V)
    dV_dc = 1j * (ZZ @ V)

    # Derivatives for left-side 1q factors
    s1q = 0.5
    dKl_dt = (-1j * s1q) * (Z @ Kl)
    dKl_dp = Rz_1l @ ((-1j * s1q) * (Y @ Ryl)) @ Rz_2l
    dKl_df = Rz_1l @ Ryl @ ((-1j * s1q) * (Z @ Rz_2l))

    dKr_dt = (-1j * s1q) * (Z @ Kr)
    dKr_dp = Rz_1r @ ((-1j * s1q) * (Y @ Ryr)) @ Rz_2r
    dKr_df = Rz_1r @ Ryr @ ((-1j * s1q) * (Z @ Rz_2r))

    # Assemble dU/dθ blocks
    derivs: list[jnp.ndarray] = []



    # Entangler parameters
    derivs.append(B @ dV_da)
    derivs.append(B @ dV_db)
    derivs.append(B @ dV_dc)

    # left/right-local parameters (Kl ⊗ Kr)
    derivs.append(kron(dKl_dt, Kr) @ V)
    derivs.append(kron(dKl_dp, Kr) @ V)
    derivs.append(kron(dKl_df, Kr) @ V)

    derivs.append(kron(Kl, dKr_dt) @ V)
    derivs.append(kron(Kl, dKr_dp) @ V)
    derivs.append(kron(Kl, dKr_df) @ V)

    # Backprop each derivative into a scalar gradient
    grads = [
        _backprop_hst_loss(L, dL_dG, dU_dth, n_sites, is_normalized)
        for dU_dth in derivs
    ]
    return jnp.stack(grads)

# TODO: for future implementations : no need to 'compose' the gate, just pass it as paramter to the _param_grad function. Since this relies on adam.py from another module, we will not do this here. 
#       however, this is a serious issue that will need attention.
def param_grad_single(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    
    dtype = dL_dG.dtype
    X, Y, Z, _ = _paulis_1q(dtype)
    kron = jnp.kron
    t, p, f = theta

    Rz_1 = _rot_from_generator(t, Z, 0.5, dtype)
    Ry   = _rot_from_generator(p, Y, 0.5, dtype)
    Rz_2 = _rot_from_generator(f, Z, 0.5, dtype)
    R = _compose_k_from_zyz(t, p, f, dtype=dtype)
    
    s1q = 0.5
    dK_dth = (-1j * s1q) * (Z @ R)
    dK_dph = Rz_1 @ ((-1j * s1q) * (Y @ Ry)) @ Rz_2
    dK_dlam = Rz_1 @ Ry @ ((-1j * s1q) * (Z @ Rz_2))

    # Assemble dU/dθ blocks
    derivs: list[jnp.ndarray] = []

    derivs.append(dK_dth)
    derivs.append(dK_dph)
    derivs.append(dK_dlam)

    # Backprop each derivative into a scalar gradient
    grads = [
        _backprop_hst_loss(L, dL_dG, dU_dth, n_sites, is_normalized)
        for dU_dth in derivs
    ]
    return jnp.stack(grads)
