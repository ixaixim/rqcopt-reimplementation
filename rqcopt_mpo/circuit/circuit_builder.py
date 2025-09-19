import rqcopt_mpo.jax_config
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional
from qiskit.synthesis import TwoQubitWeylDecomposition
from scipy.linalg import expm
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit



# ---------------------------------------------------------------------
def _random_unitary(dim: int, *, key, dtype=jnp.complex64) -> jnp.ndarray:
    """Generate a Haar‑random unitary of size (dim, dim)."""
    real_dtype = jnp.float32 if dtype==jnp.complex64 else jnp.float64

    mat = jax.random.normal(key, (dim, dim), dtype=real_dtype) + 1j * jax.random.normal(key, (dim, dim), dtype=real_dtype)
    # QR → Q is unitary, R has positive diagonal up to phases → normalise
    q, r = jnp.linalg.qr(mat) # note: q is not yet Haar-uniform. It is biased toward the phases of R. We need to absorb those phases.
    phases = jnp.diag(r) / jnp.abs(jnp.diag(r))
    return (q * phases).astype(dtype)

def generate_random_circuit(
    n_sites: int,
    n_layers: int,
    p_single: float = 0.3,
    p_two: float = 0.3,
    *,
    seed: None,
    rng: Optional[np.random.Generator] = None,
    gate_name_single: str = "U1",
    gate_name_two: str = "U2",
    dtype = jnp.complex64
) -> "Circuit":
    """
    Build a random Circuit with `n_layers` layers and `n_sites` qubits.

    At each *site* in each layer, the probabilities are:
        p_single → place a single‑qubit gate on that site
        p_two    → place a 2‑qubit gate on (site, site+1)  (ignored at last site)
        1−p_single−p_two → leave site empty

    Gates never overlap inside a layer.

    Parameters
    ----------
    n_sites, n_layers : int
        Size of the circuit.
    p_single, p_two : float
        Probabilities (per site) of generating 1‑ and 2‑qubit gates.
        They must satisfy 0 ≤ p_single, p_two and p_single + p_two ≤ 1.
    seed : int | None
        Fix the jax random seed for reproducibility. For unitary matrix generation.
    rng : np.random.Generator | None
        Optional NumPy RNG for reproducibility. For circuit structure.
    gate_name_single, gate_name_two : str
        Labels stored in each Gate’s ``name`` field.

    Returns
    -------
    Circuit
        A fully populated random circuit.
    """
    if n_sites < 1 or n_layers < 1:
        raise ValueError("n_sites and n_layers must be positive.")
    if p_single < 0 or p_two < 0 or p_single + p_two > 1:
        raise ValueError("Probabilities must satisfy 0 ≤ p_single,p_two ≤ 1 and p_single+p_two ≤ 1.")
    if rng == None: 
        rng = np.random.default_rng(seed=seed) # np distribution is independent distribution from jax distribution. You can use the same seed.


    key_pool  = (jax.random.split(jax.random.PRNGKey(seed), n_sites*n_layers))
    key_index = 0

    circuit = Circuit(n_sites=n_sites)          # empty circuit container

    for layer_idx in range(n_layers):
        while True:
            gates = []
            site = 0
            while site < n_sites:
                # determine gate type on this site
                if site == n_sites - 1:            # last qubit: can't host a 2‑qubit gate
                    probs = [p_single, 0.0, 1.0 - p_single]   # [single, two, none]
                else:
                    probs = [p_single, p_two, 1.0 - p_single - p_two]
                choice = rng.choice(["single", "two", "none"], p=probs)

                if choice == "single":
                    matrix = _random_unitary(2, key=key_pool[key_index], dtype=dtype)           # shape (2,2)
                    gates.append(
                        Gate(
                            matrix=matrix,
                            qubits=(site,),
                            layer_index=layer_idx,
                            name=gate_name_single,
                            params=(),
                            original_gate_qubits=None,
                            decomposition_part="full",
                        )
                    )
                    site += 1
                elif choice == "two":
                    matrix = _random_unitary(4, key=key_pool[key_index], dtype=dtype)
                    gates.append(
                        Gate(
                            matrix=matrix,
                            qubits=(site, site + 1),
                            layer_index=layer_idx,
                            name=gate_name_two,
                            params=(),
                            original_gate_qubits=None,
                            decomposition_part="full",
                        )
                    )
                    site += 2                                # skip neighbour (now part of this gate)
                else:  # "none"
                    site += 1

                key_index += 1
            
            if gates: # break only if non-empty
                break

                
        circuit.layers.append(
            GateLayer(
                layer_index=layer_idx,
                is_odd=(layer_idx % 2 == 1),
                gates=gates,
            )
        )

    return circuit


# ---------------------------------------------------------------------
def generate_random_brickwall_circuit(
    n_sites: int,
    n_layers: int,
    *,
    seed: Optional[int] = None,
    two_qubit_name: str = "U2",
    dtype = jnp.complex128,
) -> "Circuit":
    """
    Build a random brickwall Circuit with `n_layers` layers on `n_sites` qubits.

    - Even layers place disjoint 2-qubit gates on pairs (0,1), (2,3), ...
    - Odd  layers place disjoint 2-qubit gates on pairs (1,2), (3,4), ...
    - Each 2-qubit gate is a Haar-random unitary of size 4x4.

    Parameters
    ----------
    n_sites, n_layers : int
        Circuit width and depth.
    seed : int | None
        PRNG seed used for sampling the random unitaries.
    two_qubit_name : str
        Name assigned to the 2-qubit gates.
    dtype
        Complex dtype for generated matrices.

    Returns
    -------
    Circuit
        Brickwall circuit with random two-qubit unitaries.
    """
    if n_sites < 1 or n_layers < 1:
        raise ValueError("n_sites and n_layers must be positive.")

    key = jax.random.PRNGKey(0 if seed is None else int(seed))
    circuit = Circuit(n_sites=n_sites)

    for layer_idx in range(n_layers):
        gates = []
        start = 0 if (layer_idx % 2 == 0) else 1

        site = start
        while site + 1 < n_sites:
            key, subkey = jax.random.split(key)
            matrix = _random_unitary(4, key=subkey, dtype=dtype)
            gates.append(
                Gate(
                    matrix=matrix,
                    qubits=(site, site + 1),
                    layer_index=layer_idx,
                    name=two_qubit_name,
                    params=(),
                    original_gate_qubits=None,
                    decomposition_part="full",
                )
            )
            site += 2

        circuit.layers.append(
            GateLayer(
                layer_index=layer_idx,
                is_odd=(layer_idx % 2 == 1),
                gates=gates,
            )
        )

    return circuit
