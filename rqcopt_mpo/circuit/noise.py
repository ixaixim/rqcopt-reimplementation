import jax.numpy as jnp
import numpy as np
import copy
from rqcopt_mpo.utils.rotations import _compose_k_from_zxz
from rqcopt_mpo.optimization.rzz_ising_optimizer.utils import _rzz_matrix
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate

def update_gate_matrix_from_params(gate: Gate, dtype=jnp.complex128):
    """
    Update gate.matrix based on gate.params and gate.name.
    Supported names: 'Ising_field_term', 'Ising_no_field', 'Ising_hw_1rzz', 'Ising_hw_1q'.
    """
    name = gate.name
    params = gate.params
    kron = jnp.kron

    if name == "Ising_no_field" or name == "Ising_hw_1rzz":
        # params: (angle,) or (angle, phase)
        angle = params[0]
        if hasattr(angle, "item"):
            angle = angle.item()
        
        mat = _rzz_matrix(angle, dtype=dtype)
        
        if len(params) > 1:
            phase = params[1]
            if hasattr(phase, "item"):
                phase = phase.item()
            mat = mat * jnp.exp(1j * phase)
            
        gate.matrix = np.array(mat)

    elif name == "Ising_hw_1q":
        # params: (euler_angles,) where euler_angles is [phi, theta, lam]
        if len(params) == 0:
            raise ValueError(f"Ising_hw_1q gate has no params: {gate}")
        p = params[0]
        # In QNG branch, we use ZXZ: Rz(phi) Rx(theta) Rz(lam)
        # _compose_k_from_zxz(th, psi, ph) -> Rz(th) Rx(psi) Rz(ph)
        # So we use p[0], p[1], p[2]
        K = _compose_k_from_zxz(p[0], p[1], p[2], dtype=dtype)
        gate.matrix = np.array(K)

    elif name == "Ising_synthesized":
        # params: tuple of arrays (one per sub-gate)
        # params_dict['metadata']: list of {'name', 'qubits'}
        meta_list = gate.params_dict["metadata"]
        U_total = jnp.eye(4, dtype=dtype)
        
        for i, meta in enumerate(meta_list):
            op_name = meta["name"]
            q_indices = meta["qubits"]
            p = params[i]
            
            if op_name == "global_phase":
                # Apply global phase
                U_total = U_total * jnp.exp(1j * p[0])
                continue

            if op_name == "u3":
                # u3(theta, phi, lam) -> Rz(phi) Ry(theta) Rz(lam)
                # Keep ZYZ for u3 for backward compatibility with Qiskit default transpile if needed,
                # but if we used ZXZ in transpile, we might have rx, rz instead.
                from rqcopt_mpo.utils.rotations import _compose_k_from_zyz
                K_2x2 = _compose_k_from_zyz(p[1], p[0], p[2], dtype=dtype)
                if q_indices[0] == 1: # Left
                    U_step = kron(K_2x2, jnp.eye(2, dtype=dtype))
                else: # Right
                    U_step = kron(jnp.eye(2, dtype=dtype), K_2x2)
            elif op_name == "rx":
                # Rx(theta)
                from rqcopt_mpo.utils.rotations import _rot_from_generator, _paulis_1q
                X, _, _, _ = _paulis_1q(dtype)
                K_2x2 = _rot_from_generator(p[0], X, 0.5, dtype)
                if q_indices[0] == 1:
                    U_step = kron(K_2x2, jnp.eye(2, dtype=dtype))
                else:
                    U_step = kron(jnp.eye(2, dtype=dtype), K_2x2)
            elif op_name == "rz":
                # Rz(theta)
                from rqcopt_mpo.utils.rotations import _rot_from_generator, _paulis_1q
                _, _, Z, _ = _paulis_1q(dtype)
                K_2x2 = _rot_from_generator(p[0], Z, 0.5, dtype)
                if q_indices[0] == 1:
                    U_step = kron(K_2x2, jnp.eye(2, dtype=dtype))
                else:
                    U_step = kron(jnp.eye(2, dtype=dtype), K_2x2)
            elif op_name == "rzz":
                U_step = _rzz_matrix(p[0], dtype=dtype)
            else:
                raise ValueError(f"Unsupported sub-gate in Ising_synthesized: {op_name}")
            
            U_total = U_step @ U_total
        
        gate.matrix = np.array(U_total)

    elif name == "Ising_field_term":
        # Expected params: 8 blocks
        # K_L_upper(3), K_L_lower(3), Rzz1(1), mid_upper(3), mid_lower(3), Rzz2(1), K_R_upper(3), K_R_lower(3)
        if len(params) != 8:
             raise ValueError(f"Ising_field_term gate expects 8 parameter blocks, got {len(params)}")
        
        k_l_u = params[0]
        k_l_l = params[1]
        rzz1_ang = params[2].item()
        mid_u = params[3]
        mid_l = params[4]
        rzz2_ang = params[5].item()
        k_r_u = params[6]
        k_r_l = params[7]

        # Use ZXZ for QNG branch
        Ku_L = _compose_k_from_zxz(k_l_u[0], k_l_u[1], k_l_u[2], dtype=dtype)
        Kl_L = _compose_k_from_zxz(k_l_l[0], k_l_l[1], k_l_l[2], dtype=dtype)
        K_L = kron(Ku_L, Kl_L)

        RZZ1 = _rzz_matrix(rzz1_ang, dtype=dtype)

        Ku_M = _compose_k_from_zxz(mid_u[0], mid_u[1], mid_u[2], dtype=dtype)
        Kl_M = _compose_k_from_zxz(mid_l[0], mid_l[1], mid_l[2], dtype=dtype)
        K_M = kron(Ku_M, Kl_M)

        RZZ2 = _rzz_matrix(rzz2_ang, dtype=dtype)

        Ku_R = _compose_k_from_zxz(k_r_u[0], k_r_u[1], k_r_u[2], dtype=dtype)
        Kl_R = _compose_k_from_zxz(k_r_l[0], k_r_l[1], k_r_l[2], dtype=dtype)
        K_R = kron(Ku_R, Kl_R)

        gate.matrix = np.array(K_R @ RZZ2 @ K_M @ RZZ1 @ K_L)

def apply_parameter_noise(circuit: Circuit, std_dev: float, seed: int = None) -> Circuit:
    """
    Adds Gaussian noise to gate.params and updates gate.matrix.
    Only gates with params and a supported name are updated.
    """
    if seed is not None:
        np.random.seed(seed)
    
    noisy_circuit = circuit.copy()
    dtype = getattr(noisy_circuit, "dtype", jnp.complex128)

    for layer in noisy_circuit.layers:
        for gate in layer.gates:
            if not gate.params:
                continue
            
            # Add noise to each parameter block
            new_params = []
            for p_block in gate.params:
                noise = np.random.normal(0, std_dev, size=p_block.shape)
                new_params.append(p_block + noise)
            
            gate.params = tuple(new_params)
            
            # Update matrix if name is supported
            try:
                update_gate_matrix_from_params(gate, dtype=dtype)
            except ValueError:
                # If name not supported, matrix remains unchanged
                print("Did not apply successfully")
                
    return noisy_circuit
