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
    _d_rot_from_generator
)


def param_grad_rzz_ising_field(
    theta: jnp.ndarray,
    dL_dG: jnp.ndarray,
    L: jnp.ndarray,
    meta: dict,
    n_sites: int,
    is_normalized: bool,
) -> jnp.ndarray:
    """
    Chain rule for the Rzz TFIM parameterized circuit. Component with the field.
    The circuit operation is:
    A ⊗ B Rzz2 C ⊗ D Rzz1 E ⊗ F

    in ASCII notation: 
     ┌────────────┐             ┌─────────┐              ┌────────────┐
q_0: ┤ E          ├─■───────────┤ C       ├─■────────────┤ A          ├
     ├────────────┤ │ZZ(      ) ├─────────┤ │ZZ(       ) ├────────────┤
q_1: ┤ F          ├─■───────────┤ D       ├─■────────────┤ B          ├
     └────────────┘             └─────────┘              └────────────┘
    where A,B,C,D,E,F are arbitrary unitaries (here decomposed as ZYZ angles)
    """
    dtype = dL_dG.dtype
    _, Y, Z, I2 = _paulis_1q(dtype)
    kron = jnp.kron

    # unpack parameters: 18 params
    E_th, E_ph, E_lam = theta[0:3]
    Rzz1 = theta[3]
    # todo: ...
    
    # rotation helpers
    rot_z = lambda ang: _rot_from_generator(ang, Z, 0.5, dtype)
    rot_y = lambda ang: _rot_from_generator(ang, Y, 0.5, dtype)
    drot_z = lambda ang: _d_rot_from_generator(ang, Z, 0.5, dtype)
    drot_y = lambda ang: _d_rot_from_generator(ang, Y, 0.5, dtype)

    # dressing single-qubit blocks left side
    Rz1_E = rot_z(E_ph)
    Ry_E = rot_z(E_th)
    Rz2_E = rot_z(E_lam)
    # todo F
    K_L_upper = Rz1_E @ Ry_E @ Rz2_E
    K_L_lower = 
    dressing = kron(K_L_upper, K_L_lower)

    derivs: list[jnp.ndarray] = []
    # derivative pieces for dressings



