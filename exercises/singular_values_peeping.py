import rqcopt_mpo.jax_config

from pathlib import Path

import jax.numpy as jnp
import numpy as np

# MPO builder and circuits
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import rzz_decompose_ising_circuit #hw friendly
from rqcopt_mpo.optimization.rzz_ising_optimizer.rzz_ising_optimizer_hw_friendly import optimize # hw_friendly
# from rqcopt_mpo.optimization.utils import overlap_to_loss
# from rqcopt_mpo.optimization.adam_utils import make_early_stop
# from rqcopt_mpo.optimization.cnot_optimizer.utils import 
# from experiments.utils import save_data_npz


n_sites = 10 # choose even number
J = 0. # HAS TO BE ZERO
D = 1.
hx, hz = 0.75, 0.6
t = 2.0 # time of evolution

if J != 0 or hx==0 or hz==0:
    raise ValueError("Only Transverse Field Ising Model (TFIM) is permitted here.")

reps = 10
order = 4 
dt = t/reps
dtype = jnp.complex128
target_is_normalized = False 



# patience = 10
# min_delta = 1e-8
# early_stop = make_early_stop(patience=patience, min_delta=min_delta)



target_circ = trotterized_hardware_friendly_xyz_circuit(    
    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz,
    order=order, dt=dt, reps=reps, collapse=True,
    dtype=dtype
)
print(f"Target circuit with {target_circ.num_layers} layers")
target_mpo = circuit_to_mpo(target_circ, svd_cutoff=svd_cutoff)

print("End program")
