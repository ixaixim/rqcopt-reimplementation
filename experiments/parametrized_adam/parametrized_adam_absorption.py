import rqcopt_mpo.jax_config
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# circuit utils
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.circuit.decompose.single_q_decompose import euler_zyz_decompose_circuit
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import (
    weyl_decompose_circuit,
    absorb_single_qubit_layers,
)

# MPO conversion
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo

# optimizer
from rqcopt_mpo.optimization.parametrized_adam.optimizer import optimize
from rqcopt_mpo.optimization.utils import overlap_to_loss

# save utilities
from experiments.utils import save_data_npz


def main():

    # Build target circuit (random small circuit)
    # trotterization params (match weyl optimizer comparison experiment)
    n_sites = 6  # choose even number
    J = 1.0
    D = -1.0
    h = 0.0
    t = 0.5
    reps = 10
    order = 4
    dt = t / reps
    dtype = jnp.complex128
    target_is_normalized = False

    # set up target MPO
    target_circ = trotterized_heisenberg_circuit(
        n_sites=n_sites, J=J, D=D, h=h,
        order=order, dt=dt, reps=reps,
        dtype=dtype
    )
    print(f"Target circuit with {target_circ.num_layers} layers")
    target_mpo = circuit_to_mpo(target_circ)
    target_mpo.left_canonicalize(normalize=target_is_normalized)

    # set up quantum circuit
    reps = 3
    dt = t / reps
    init_circ = trotterized_heisenberg_circuit(
        n_sites=n_sites,
        J=J,
        D=D,
        dt=dt,
        reps=reps,
        order=2,
        dtype=jnp.complex128,
    )
    print(f"Initial circuit with {init_circ.num_layers} layers")
    print(f"Initial Fidelity of Circuit: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")
    
    # Pipeline: Weyl → absorb single-qubit layers → Euler ZYZ for 1q gates
    init_circ = weyl_decompose_circuit(init_circ, keep_global_phase=False)
    init_circ = absorb_single_qubit_layers(init_circ)
    init_circ = euler_zyz_decompose_circuit(init_circ, include_global_phase=False)
    print(f"Init Circuit Decomposed with {init_circ.num_layers} layers")
    print(f"Initial Fidelity: {overlap_to_loss(np.trace(init_circ.to_matrix().conjugate().T @ target_circ.to_matrix()), n_sites=n_sites, normalize=target_is_normalized)}")
    
    # Optimization parameters
    lr = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    clip_grad_norm = None
    max_steps = 200
    max_bondim_env = 128
    svd_cutoff = 0.0

    # Run parametrized Adam
    loss = optimize(
        init_circ,
        target_mpo,
        lr=lr,
        betas=betas,
        eps=eps,
        clip_grad_norm=clip_grad_norm,
        max_steps=max_steps,
        max_bondim_env=max_bondim_env,
        svd_cutoff=svd_cutoff,
    )

    # Save loss curve
    base_dir = Path(__file__).resolve().parent
    save_data_npz(base_dir, "loss_parametrized_adam_absorb", loss, method="Parametrized_Adam_absorb")


if __name__ == "__main__":
    main()
