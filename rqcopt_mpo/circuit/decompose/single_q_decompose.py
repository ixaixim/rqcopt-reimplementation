import rqcopt_mpo.jax_config

from typing import Dict, List, Tuple
import numpy as np
from qiskit.synthesis import OneQubitEulerDecomposer
from rqcopt_mpo.circuit.circuit_builder import _random_unitary

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit

def _Rz(a: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j*a), 0.0j],
                     [0.0j, np.exp(0.5j*a)]], dtype=np.complex128)

def _Ry(b: float) -> np.ndarray:
    c, s = np.cos(b/2.0), np.sin(b/2.0)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.complex128)


def zyz_decompose_gate(gate: Gate, out_layer: int) -> tuple[float, list[Gate]]:
    """
    Decompose a 1-qubit Gate into ZYZ rotations and return:
        (phase, [Rz(lam) @ Ry(theta) @ Rz(phi)])
    where `phase` is a global phase such that exp(1j*phase) * product(parts) == gate.matrix.

    For non-1q gates, returns (0.0, [gate moved to out_layer]).
    """
    if not gate.is_single_qubit():
        g = gate.copy()
        g.layer_index = out_layer
        return 0.0, [g]

    # --- 1) Angles from Qiskit (ZYZ basis) -------------------------------
    decomp = OneQubitEulerDecomposer(basis="ZYZ")
    theta, phi, lam, phase = decomp.angles_and_phase(np.asarray(gate.matrix, dtype=np.complex128))
    q = gate.qubits[0]

    parts: list[Gate] = []

    # naming: Gate = Rz2(phi) @ Ry(theta) @ Rz1(lam), NOTE: in circuit Rz1 comes first
    
    # Rz(lam)
    G_rz1 = Gate(
        matrix=_Rz(lam),
        qubits=(q,),
        layer_index=out_layer,
        name="RZ",
        decomposition_part="Rz1",
        params=(lam,),
        original_gate_qubits=gate.qubits
    )
    parts.append(G_rz1)

    # Ry(theta)
    G_ry = Gate(
        matrix=_Ry(theta),
        qubits=(q,),
        layer_index=G_rz1.layer_index + 1,
        name="RY",
        decomposition_part="Ry",
        params=(theta,),
        original_gate_qubits=gate.qubits
    )
    parts.append(G_ry)

    # Rz(phi)
    G_rz2 = Gate(
        matrix=_Rz(phi),
        qubits=(q,),
        layer_index=G_ry.layer_index + 1,
        name="RZ",
        decomposition_part="Rz2",
        params=(phi,),
        original_gate_qubits=gate.qubits
    )
    parts.append(G_rz2)



    return phase, parts


# take circuit, decompose single qubit gates, return new circuit where gates have been decomposed. 
def euler_zyz_decompose_circuit(orig: Circuit, *, include_global_phase: bool = False) -> Circuit:
    """
    Return a new Circuit where every single-qubit gate is replaced by its
    ZYZ Euler rotations, each on its own (new) layer. Two-qubit gates are
    shifted so they do not overlap with the new rotations.

    Layer mapping strategy (per original layer L):
        base = L * stride
        - 1q gates -> (Phase?) + Rz + Ry + Rz at layers base, base+1, base+2, (base+3)
        - 2q gates -> placed at base + R  (R = 3 if no phase, else 4)
    Different original layers are assigned disjoint blocks, so no cross-block conflicts.

    Args:
        orig: input Circuit
        include_global_phase: if True, emit an explicit global phase 1q gate
                              before the rotations (adds one extra layer).

    Returns:
        A new Circuit with expanded, non-overlapping layers.
    """
    # Hardcoded assumption:
    # assume only layers of just single qubit gates, or just two qubit gates, NO MIXTURE
    
    R = 3  # number of layers used by each 1q decomposition
    last_idx_saved = -1
    new_layers: Dict[int, GateLayer] = {}

    for layer in sorted(orig.layers, key=lambda L: L.layer_index):
        kinds = { (g.is_single_qubit(), g.is_two_qubit()) for g in layer.iterate_gates(False) }
        assert len(kinds) == 1, "Layer mixes 1q and 2q gates, which is unsupported."
        is_1q_layer = next(iter(kinds))[0]

        if is_1q_layer:
            start = last_idx_saved + 1
            # prepare the R output layers
            for r in range(R):
                idx = start + r
                if idx not in new_layers:
                    new_layers[idx] = GateLayer(
                        layer_index=idx,
                        is_odd=layer.is_odd, # should drop layer is odd, is meaningless here
                        gates=[],
                        n_sites=orig.n_sites
                    )

            # Decompose each 1q gate and place its r-th piece on layer start+r
            for gate in layer.iterate_gates(False):
                phase, pieces = zyz_decompose_gate(
                    gate,
                    out_layer=start,  # we set per-piece below
                )
                pieces_sorted = sorted(pieces, key=lambda g: g.layer_index)
                assert len(pieces) == R, "Decomposer must return exactly R pieces."
                for r, pg in enumerate(pieces_sorted):
                    idx = start + r
                    new_layers[idx].add_gate(pg)

            last_idx_saved = start + R - 1 # TODO: check if correct

        else:
            # 2-qubit layer (single output layer)
            idx = last_idx_saved + 1
            if idx not in new_layers:
                new_layers[idx] = GateLayer(
                    layer_index=idx,
                    is_odd=layer.is_odd, # can be removed
                    gates=[],
                    n_sites=orig.n_sites
                )
            for gate in layer.iterate_gates(False):
                new_gate = gate.copy()
                new_gate.layer_index = idx
                new_layers[idx].add_gate(new_gate)

            last_idx_saved += 1

    layers = [new_layers[idx] for idx in sorted(new_layers)]
    final = Circuit(n_sites=orig.n_sites, dtype=orig.dtype, layers=layers)
    return final





