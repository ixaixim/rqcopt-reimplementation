import rqcopt_mpo.jax_config
import jax.numpy as jnp
import numpy as np

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from qiskit.synthesis import OneQubitEulerDecomposer
from rqcopt_mpo.utils.rotations import (
    _backprop_hst_loss,
    _compose_k_from_zyz,
    _paulis_1q,
    _rot_from_generator,
)


def _update_circuit_from_trees(circuit, params_tree, meta_tree) -> None:
    """
    Rewrite gate parameters/matrices using the updated parameter tree.
    Supports absorbed CNOT blocks (name \"Abs\") and the leading single-qubit layer (\"1qLayer0\").
    """
    dtype = getattr(circuit, "dtype", jnp.complex128)
    circuit.sort_layers()
    _, Y, Z, I2 = _paulis_1q(dtype)
    kron = jnp.kron
    C01, C10 = _cnot_matrices(dtype)

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
            if name == "CNOT_block":
                if len(pieces) != 5:
                    raise ValueError(f"Abs gate expects 5 parameter blocks, got {len(pieces)}")
                d = jnp.asarray(pieces[0], dtype=jnp.float64).item()
                b = jnp.asarray(pieces[1], dtype=jnp.float64).item()
                a = jnp.asarray(pieces[2], dtype=jnp.float64).item()
                upper = jnp.asarray(pieces[3], dtype=jnp.float64)
                lower = jnp.asarray(pieces[4], dtype=jnp.float64)

                Ku = _compose_k_from_zyz(upper[0], upper[1], upper[2], dtype=dtype)
                Kl = _compose_k_from_zyz(lower[0], lower[1], lower[2], dtype=dtype)

                Rz0 = _rot_from_generator(d, Z, 0.5, dtype)
                Ry_b = _rot_from_generator(b, Y, 0.5, dtype)
                Ry_a = _rot_from_generator(a, Y, 0.5, dtype)

                middle = (
                    C10
                    @ kron(I2, Ry_a)
                    @ C01
                    @ kron(I2, Ry_b)
                    @ kron(Rz0, I2)
                    @ C10
                )
                gate.matrix = kron(Ku, Kl) @ middle
                gate.params = (
                    jnp.asarray(pieces[0], dtype=jnp.float64),
                    jnp.asarray(pieces[1], dtype=jnp.float64),
                    jnp.asarray(pieces[2], dtype=jnp.float64),
                    upper,
                    lower,
                )
            elif name == "1qLayer0":
                single_params = jnp.asarray(pieces[0], dtype=jnp.float64)
                gate.matrix = _compose_k_from_zyz(single_params[0], single_params[1], single_params[2], dtype=dtype)
                gate.params = (single_params,)
            else:
                gate.params = tuple(jnp.asarray(p, dtype=jnp.float64) for p in pieces)


def parametrize_circuit(circ: Circuit) -> Circuit:
    """
    Parametrizes an absorbed CNOT decomposed circuit  (assumes gates are already grouped together for faster MPO sweeping).
    Mutate the circuit in place. Appends parameters to each gate.
    each 2q gate consists of a structure: as such:
    (Pennylane structure) NOTE: the first two 1q gates (A,B) have been absorbed. 
     +---+          +-------+                    +---+
q_0: | A |--(+)-----| Rz(a) |----*-----------(+)--| C |--
     +---+   |      +-------+    |            |   +---+
     +---+   |      +-------+    |  +-------+ |   +---+
q_1: | B |---*------| Ry(b) |---(+)--| Ry(c) |-*--| D |--
     +---+          +-------+        +-------+    +---+

     After absorption (9 free parameters):
            +-------+                    +---+
q_0:(+)-----| Rz(a) |----*-----------(+)--| C |--
     |      +-------+    |            |   +---+
     |      +-------+    |  +-------+ |   +---+
q_1: *------| Ry(b) |---(+)--| Ry(c) |-*--| D |--
            +-------+        +-------+    +---+

    Parameter layout: (a,b,c, C_param, D_param, )
    """ 
    decomp = OneQubitEulerDecomposer(basis="ZYZ")

    # LAYER 0
    # keep the leading single-qubit layer untouched and add Euler parameters to it
    for single_qubit_gate in circ.layers[0].iterate_gates():
        th, ph, lam, _ = decomp.angles_and_phase(single_qubit_gate.matrix)
        single_params = jnp.array([ph, th, lam], dtype=jnp.float64)
        single_qubit_gate.name = "1qLayer0"
        single_qubit_gate.params += (single_params,)

    # ALL OTHER LAYERS
    for pair_idx, two_qubit_layer in enumerate(circ.layers[1:]):
        for gate in two_qubit_layer.gates:
            # params for rotations
            gate.params += (np.array(gate.params_dict["RZ"][0]),)
            gate.params +=  (np.array(gate.params_dict["RY"][0]),)
            gate.params +=  (np.array(gate.params_dict["RY"][1]),)

            # params for C and D matrices
            for label in ("upper", "lower"):
                th, ph, lam, _ = decomp.angles_and_phase(gate.params_dict[label])
                params = jnp.array([ph, th, lam], dtype=jnp.float64)
                gate.params += (params,)

    return circ


def _d_rot_from_generator(theta, P, scale, dtype):
    """Derivative of _rot_from_generator wrt theta."""
    c = jnp.cos(scale * theta)
    s = jnp.sin(scale * theta)
    I = jnp.eye(P.shape[0], dtype=dtype)
    return -scale * s * I - 1j * scale * c * P


def _cnot_matrices(dtype):
    """Return (CNOT_01, CNOT_10) using (q0, q1) ordering."""
    cnot_01 = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=dtype,
    )
    cnot_10 = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=dtype,
    )
    return cnot_01, cnot_10


def param_grad_cnot_abs(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Chain rule for the absorbed 3-CNOT block with nine parameters:
    (d, b, a, upper_phi, upper_theta, upper_lambda, lower_phi, lower_theta, lower_lambda), where
    `d` is the RZ angle on qubit 0, `b` is the first RY on qubit 1, and `a` is the second RY on qubit 1.
    The middle circuit operation is:
    C ⊗ D (CNOT(1->0) I ⊗ RY(a) CNOT(0->1) ⊗ RY(b) RZ(d) CNOT(1->0))
    

    in ASCII notation:
            +-------+                     +---+
q_0:(+)-----| Rz(d) |----*-----------(+)--| C |--
     |      +-------+    |            |   +---+
     |      +-------+    |  +-------+ |   +---+
q_1: *------| Ry(b) |---(+)--| Ry(a) |-*--| D |--
            +-------+        +-------+    +---+

    dressed by kron(K_upper, K_lower) where each K is ZYZ.
    """
    dtype = dL_dG.dtype
    _, Y, Z, I2 = _paulis_1q(dtype)
    kron = jnp.kron

    # unpack parameters
    d, b, a = theta[0], theta[1], theta[2]
    up_phi, up_th, up_lam = theta[3:6]
    lo_phi, lo_th, lo_lam = theta[6:9]

    # rotation helpers
    rot_z = lambda ang: _rot_from_generator(ang, Z, 0.5, dtype)
    rot_y = lambda ang: _rot_from_generator(ang, Y, 0.5, dtype)
    drot_z = lambda ang: _d_rot_from_generator(ang, Z, 0.5, dtype)
    drot_y = lambda ang: _d_rot_from_generator(ang, Y, 0.5, dtype)

    # dressing single-qubit blocks
    Rz_up_1 = rot_z(up_phi)
    Ry_up = rot_y(up_th)
    Rz_up_2 = rot_z(up_lam)
    Ku = Rz_up_1 @ Ry_up @ Rz_up_2

    Rz_lo_1 = rot_z(lo_phi)
    Ry_lo = rot_y(lo_th)
    Rz_lo_2 = rot_z(lo_lam)
    Kl = Rz_lo_1 @ Ry_lo @ Rz_lo_2

    # derivative pieces for dressings
    dKu_dphi = drot_z(up_phi) @ Ry_up @ Rz_up_2
    dKu_dth = Rz_up_1 @ drot_y(up_th) @ Rz_up_2
    dKu_dlam = Rz_up_1 @ Ry_up @ drot_z(up_lam)

    dKl_dphi = drot_z(lo_phi) @ Ry_lo @ Rz_lo_2
    dKl_dth = Rz_lo_1 @ drot_y(lo_th) @ Rz_lo_2
    dKl_dlam = Rz_lo_1 @ Ry_lo @ drot_z(lo_lam)

    # central rotations 
    Rz0 = rot_z(d)
    Ry_b = rot_y(b)
    Ry_a = rot_y(a)

    dRz0 = drot_z(d)
    dRy_b = drot_y(b)
    dRy_a = drot_y(a)

    C01, C10 = _cnot_matrices(dtype)

    # fixed identities
    I = I2

    # middle block U_mid = C10 @ (I⊗RY_a) @ C01 @ (I⊗RY_b) @ (RZ⊗I) @ C10
    U_mid = (
        C10
        @ kron(I, Ry_a)
        @ C01
        @ kron(I, Ry_b)
        @ kron(Rz0, I)
        @ C10
    )
    dressing = kron(Ku, Kl)

    derivs: list[jnp.ndarray] = []

    # d (RZ on qubit 0)
    derivs.append(
        dressing
        @ C10
        @ kron(I, Ry_a)
        @ C01
        @ kron(I, Ry_b)
        @ kron(dRz0, I)
        @ C10
    )

    # b (first RY on qubit 1)
    derivs.append(
        dressing
        @ C10
        @ kron(I, Ry_a)
        @ C01
        @ kron(I, dRy_b)
        @ kron(Rz0, I)
        @ C10
    )

    # a (second RY on qubit 1)
    derivs.append(
        dressing
        @ C10
        @ kron(I, dRy_a)
        @ C01
        @ kron(I, Ry_b)
        @ kron(Rz0, I)
        @ C10
    )

    # upper dressing (ZYZ)
    derivs.append(kron(dKu_dphi, Kl) @ U_mid)
    derivs.append(kron(dKu_dth, Kl) @ U_mid)
    derivs.append(kron(dKu_dlam, Kl) @ U_mid)

    # lower dressing (ZYZ)
    derivs.append(kron(Ku, dKl_dphi) @ U_mid)
    derivs.append(kron(Ku, dKl_dth) @ U_mid)
    derivs.append(kron(Ku, dKl_dlam) @ U_mid)

    grads = [
        _backprop_hst_loss(L, dL_dG, dU_dth, n_sites, is_normalized)
        for dU_dth in derivs
    ]
    return jnp.stack(grads)
