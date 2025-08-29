import rqcopt_mpo.jax_config

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Optional
from qiskit.synthesis import TwoQubitWeylDecomposition
from scipy.linalg import expm
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple

# TODO: this implementation currently is not implemented in JAX.
def weyl_decompose_gate(gate: Gate, k2_layer: int, keep_global_phase: bool = True) -> list[Gate]:
    """
    Replace a two-qubit Gate by its Weyl factors.
    Returns three **new** Gate objects whose `layer_index`
    are k2_layer, k2_layer+1, k2_layer+2.
    Single-qubit gates are returned unchanged.
    """
    if not gate.is_two_qubit():
        # Nothing to do – keep it in its original layer.
        g = gate.copy()
        g.layer_index = k2_layer
        return [g]

    # --- 1 : call Qiskit
    decomp = TwoQubitWeylDecomposition(gate.matrix)
    k1l, k1r, k2l, k2r = decomp.K1l, decomp.K1r, decomp.K2l, decomp.K2r
    a, b, c = decomp.a, decomp.b, decomp.c
    gl_phase = np.exp(1j * decomp.global_phase) # not essential for compression purposes, but used to test the match of the weyl decomposed circuit with the original circuit

    # --- 2 : build the three factors ---------------------------------------
    left, right = sorted(gate.qubits)           # (min, max)

    # (a) K2 layer -----------------------------------------------------------
    g_k2l = Gate(matrix=k2l, qubits=(left,),  layer_index=k2_layer,
                 name="K2l", decomposition_part="K2l",
                 original_gate_qubits=gate.qubits)
    g_k2r = Gate(matrix=k2r, qubits=(right,), layer_index=k2_layer,
                 name="K2r", decomposition_part="K2r",
                 original_gate_qubits=gate.qubits)

    # (b) e^{i(...)} layer ---------------------------------------------------
    XX = np.kron([[0,1],[1,0]], [[0,1],[1,0]])
    YY = np.kron([[0,-1j],[1j,0]], [[0,-1j],[1j,0]])
    ZZ = np.kron([[1,0],[0,-1]], [[1,0],[0,-1]])
    nonlocal_op = gl_phase * expm(1j*(a*XX + b*YY + c*ZZ))

    params_nonlocal_op = (a,b,c,decomp.global_phase) if keep_global_phase else (a,b,c)
    
    g_ent  = Gate(matrix=nonlocal_op,
                  qubits=(left, right),
                  layer_index=k2_layer+1,
                  name="Exp(XX+YY+ZZ)", decomposition_part="Exp",
                  params=params_nonlocal_op,
                  original_gate_qubits=gate.qubits)

    # (c) K1 layer -----------------------------------------------------------
    g_k1l = Gate(matrix=k1l, qubits=(left,),  layer_index=k2_layer+2,
                 name="K1l", decomposition_part="K1l",
                 original_gate_qubits=gate.qubits)
    g_k1r = Gate(matrix=k1r, qubits=(right,), layer_index=k2_layer+2,
                 name="K1r", decomposition_part="K1r",
                 original_gate_qubits=gate.qubits)

    return [g_k2l, g_k2r, g_ent, g_k1l, g_k1r]

def weyl_decompose_circuit(orig: Circuit, keep_global_phase=True) -> Circuit:
    """
    Map an n-layer brickwall Circuit to a 3 n-layer Circuit
    by Weyl-decomposing every two-qubit gate.
    """
    new_layers: dict[int, GateLayer] = {}

    # layer_offset grows by +2 after every *original* layer
    for layer in sorted(orig.layers, key=lambda L: L.layer_index):
        for gate in layer.gates:
            # Put K2 factors at 'layer.layer_index + layer_offset = target_base'
            target_base = layer.layer_index * 3       
            pieces = weyl_decompose_gate(gate, target_base, keep_global_phase=keep_global_phase)

            # drop each piece into the correct GateLayer --------------------
            for pg in pieces:
                idx = pg.layer_index
                if idx not in new_layers:
                    new_layers[idx] = GateLayer(layer_index=idx,
                                                is_odd=layer.is_odd,  # parity is arbitrary now
                                                gates=[], n_sites=orig.n_sites)
                new_layers[idx].add_gate(pg)

    # build the final Circuit in layer order
    final = Circuit(n_sites=orig.n_sites,
                    layers=[new_layers[idx] for idx in sorted(new_layers)],
                    hamiltonian_type=orig.hamiltonian_type,
                    trotter_params=orig.trotter_params)
    return final

def absorb_single_qubit_layers(weyl_circ: Circuit) -> Circuit:
    """
    Collapse each consecutive (K1-layer , K2-layer) pair into one layer,
    skipping boundary qubits and boundary layers.

    Input : 3 N layers (K2, Exp, K1)×N
    Output: 2 N+1 layers
    """
    weyl_circ.sort_layers()
    L_tot   = weyl_circ.num_layers            # = 3 N
    n_sites = weyl_circ.n_sites
    first_q, last_q = 0, n_sites - 1

    new_layers: dict[int, GateLayer] = {}
    new_idx = 0            # index in the *output* circuit

    # ---- helper -------------------------------------------------------
    def _clone_layer(src: GateLayer, tgt_idx: int):
        layer = GateLayer(layer_index=tgt_idx,
                          is_odd=src.is_odd,
                          n_sites=n_sites,
                          gates=[g.copy() for g in src.gates])
        for g in layer.gates:
            g.layer_index = tgt_idx
        new_layers[tgt_idx] = layer

    # ---- copy the first two layers verbatim --------------------------
    _clone_layer(weyl_circ.layers[0], new_idx); new_idx += 1   # K2
    _clone_layer(weyl_circ.layers[1], new_idx); new_idx += 1   # Exp

    i = 2   # first K1 layer in the input circuit

    while i < L_tot - 3:        # stop *before* the last K1/Exp pair
        layer_K1, layer_K2 = weyl_circ.layers[i], weyl_circ.layers[i+1]

        # lookup tables  qubit → Gate
        k1 = {g.qubits[0]: g for g in layer_K1.gates}
        k2 = {g.qubits[0]: g for g in layer_K2.gates}

        absorbed = GateLayer(layer_index=new_idx,
                             is_odd=layer_K1.is_odd,
                             n_sites=n_sites, gates=[])

        for q in sorted(set(k1)|set(k2)):
            g1, g2 = k1.get(q), k2.get(q)

            if (q in (first_q, last_q)) or (g1 is None) or (g2 is None):
                # keep them separate (put both, earliest first)
                for g in (g1, g2):
                    if g is not None:
                        gc = g.copy(); gc.layer_index = new_idx
                        absorbed.add_gate(gc)
            else:
                # true absorption :  g2 ∘ g1
                merged = Gate(matrix   = g2.matrix @ g1.matrix,
                              qubits   = g1.qubits,
                              layer_index = new_idx,
                              name     = "Abs",
                              decomposition_part = "Abs",
                              original_gate_qubits = g1.original_gate_qubits)
                absorbed.add_gate(merged)

        new_layers[new_idx] = absorbed
        new_idx += 1

        # copy the following Exp layer unchanged -----------------------
        _clone_layer(weyl_circ.layers[i+2], new_idx)
        new_idx += 1
        i += 3          # advance to the next K1 layer

    # ---- copy only the *last* single-qubit layer ---------------------
    _clone_layer(weyl_circ.layers[-1], new_idx)   # final K1

    return Circuit(n_sites=n_sites,
                   layers=[new_layers[k] for k in sorted(new_layers)],
                   hamiltonian_type=weyl_circ.hamiltonian_type,
                   trotter_params=weyl_circ.trotter_params)

