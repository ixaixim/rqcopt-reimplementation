import rqcopt_mpo.jax_config

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple

from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer

# Two-qubit basis decomposition that expands every gate into single-qubit
# rotations and CNOTs. The decomposer is global to avoid re-instantiation.
basis = CXGate()
decomposer = TwoQubitBasisDecomposer(basis)

# note: qiskit uses little endian ordering: basis states are ordere as |q1q0> for 2 qubits;
#       a gate on qubit 0 acts as I prod U.
# we just switch indices in places when returning the dictionary in _decompose_gate_into_cnot_blocks
def _embed_single_into_two(mat: np.ndarray, position: int, dtype) -> np.ndarray:
    """Embed a single-qubit matrix on the lower (0) or upper (1) qubit."""
    eye = np.eye(2, dtype=dtype)
    if position == 0:
        return np.kron(eye, mat)
    if position == 1:
        return np.kron(mat, eye)
    raise ValueError(f"Invalid qubit position {position}")


def _single_ops(
    ops, dtype: np.dtype
) -> Dict[int, np.ndarray]:
    """Collapse a sequence of 1q instructions into per-qubit matrices."""
    totals = {}
    for instr in ops:
        op = instr.operation
        if op.num_qubits != 1:
            raise ValueError("Expected single-qubit operation in this block.")
        pos = instr.qubits[0]._index # with recent qiskit version, no other way to access index (would otherwise need to pass circ and then do circ.find_bit(instr.qubits[0]).index
        totals[pos] = instr.matrix 
    return totals

def _build_two_qubit_block(
    ops, dtype: np.dtype
) -> np.ndarray:
    """Collapse a sequence containing CNOTs and 1q rotations into a single 4x4 matrix."""
    # TODO: will later also save the 
    block = np.eye(4, dtype=dtype)
    for instr in ops:
        op = instr.operation
        step = instr.matrix
        if op.num_qubits == 1:
            pos = instr.qubits[0]._index
            step = _embed_single_into_two(step, pos, dtype)
        elif op.num_qubits != 2:
            raise ValueError("Instruction acting on more than two qubits.")
        block = step @ block
    return block


def _decompose_gate_into_cnot_blocks(
    gate: Gate, basis_fidelity: Optional[float]
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
    """Return (pre, middle, post) blocks for a two-qubit gate."""
    if not gate.is_two_qubit():
        raise ValueError("CNOT decomposition expects a two-qubit gate.")

    # permute the matrix indices:

    circ = decomposer(
        Operator(np.array(gate.matrix, dtype=np.complex128)),
        basis_fidelity=basis_fidelity,
    )
    instructions = list(circ.data)
    cnot_indices = [
        idx for idx, instr in enumerate(instructions) if instr.operation.num_qubits == 2
    ]
    if not cnot_indices:
        raise ValueError("Basis decomposition did not produce any two-qubit gate.")
    first, last = cnot_indices[0], cnot_indices[-1]
    # TODO: add a check for printing: "gate on q: {gate.qubits} has {len(cnot_indices)} cnots."
    dtype = gate.matrix.dtype

    pre_ops = instructions[:first]
    middle_ops = instructions[first : last + 1]
    post_ops = instructions[last + 1 :]

    pre = _single_ops(pre_ops, dtype)
    middle = _build_two_qubit_block(middle_ops, dtype)
    post = _single_ops(post_ops, dtype)

    return (
        {gate.qubits[pos]: pre[1-pos] for pos in range(2)},
        middle,
        {gate.qubits[pos]: post[1-pos] for pos in range(2)},
    )


def _add_single_gate(
    layer: GateLayer,
    qubit: int,
    matrix: np.ndarray,
    original_qubits: Tuple[int, int],
    name: str,
):
    layer.add_gate(
        Gate(
            matrix=matrix,
            qubits=(qubit,),
            layer_index=layer.layer_index,
            name=name,
            decomposition_part=name,
            original_gate_qubits=original_qubits,
        )
    )


def _collect_next_layer_pre(
    layer: GateLayer,
    gate_decomp: Dict[int, Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]],
) -> Dict[int, np.ndarray]:
    """Return a {qubit: pre_matrix} map for the gates in the provided layer."""
    if layer is None: return None
    qubit_to_pre: Dict[int, np.ndarray] = {}
    for gate in layer.iterate_gates():
        pre, _, _ = gate_decomp[id(gate)]
        qubit_to_pre.update(pre)
    return qubit_to_pre


def cnot_decompose_circuit(
    orig: Circuit, basis_fidelity: Optional[float] = None
) -> Dict[int, Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]]:
    """
    Returns a lookup table that holds the decomposition of the gate.
    """

    n_sites = orig.n_sites
    gate_decomp = {}
    for layer in sorted(orig.layers, key=lambda L: L.layer_index):
        for gate in layer.iterate_gates():

            pre, middle, post = _decompose_gate_into_cnot_blocks(gate, basis_fidelity)

            # create a lookup table: for the gate id, associate the decomposition
            gate_decomp[id(gate)] = (pre, middle, post)
    return gate_decomp

def cnot_absorb_1q_gates(
        orig: Circuit, basis_fidelity: Optional[float] = None
) -> Circuit: 
    "Assumes brickwall circuit, starting with even layers"
    sorted_layers = sorted(orig.layers, key=lambda layer: layer.layer_index)
    n_qubits = orig.n_sites
    new_layers: dict[int, GateLayer] = {}
    new_layers[0] = GateLayer(
        layer_index=0,
        is_odd=sorted_layers[0].is_odd,
        n_sites=orig.n_sites,
        gates=[],
    )
    gate_decomp = cnot_decompose_circuit(orig, basis_fidelity)

    # save first 1q layer:
    for gate in sorted_layers[0].iterate_gates():
        pre, _, _ = gate_decomp[id(gate)]
        for qubit, matrix in pre.items():
            _add_single_gate(new_layers[0], qubit, matrix, gate.qubits, name="pre")

    last_layer = sorted_layers[-1]
    iterate_up_to_idx = -1 if not last_layer.is_odd else -2 # if last layer is odd, we manually manage the last two layers
    for idx, layer in enumerate(sorted_layers[:iterate_up_to_idx]): 
            
        next_layer = sorted_layers[idx + 1] if idx + 1 < len(sorted_layers) else None
        next_next_layer = sorted_layers[idx + 2] if idx + 2 < len(sorted_layers) else None
        next_layer_pre = _collect_next_layer_pre(next_layer, gate_decomp) # for inner qubits absorption
        next_next_layer_pre = _collect_next_layer_pre(next_next_layer, gate_decomp) # for boundary qubits absorbtion

        for gate in layer.iterate_gates():
        
            _, middle, post = gate_decomp[id(gate)]
            first, second = gate.qubits
            upper = next_layer_pre[first] @ post[first]  if first != 0 else next_next_layer_pre[first] @ post[first]
            lower = next_layer_pre[second] @ post[second] if second != n_qubits-1 else next_next_layer_pre[second] @ post[second]
            matrix = jnp.kron(upper, lower) @ middle
            target = new_layers.setdefault(
                layer.layer_index + 1,
                GateLayer(layer_index=layer.layer_index + 1,
                        is_odd=layer.is_odd,
                        n_sites=orig.n_sites,
                        gates=[]),
            )
            target.add_gate(
                Gate(
                    matrix=matrix,
                    qubits=gate.qubits,
                    layer_index=target.layer_index,
                    name="Abs",
                    decomposition_part="Abs",
                    original_gate_qubits=gate.qubits,
                )
            )
        
    # second to last layer (to manage only if last_layer = odd)
    if last_layer.is_odd:
        layer = sorted_layers[-2]
        next_layer = sorted_layers[-1]
        next_layer_pre = _collect_next_layer_pre(next_layer, gate_decomp)

        for gate in layer.iterate_gates():
            
            _, middle, post = gate_decomp[id(gate)]
            first, second = gate.qubits
            upper = next_layer_pre[first] @ post[first]  if first != 0 else post[first]
            lower = next_layer_pre[second] @ post[second] if second != n_qubits-1 else post[second]
            matrix = jnp.kron(upper, lower) @ middle
            target = new_layers.setdefault(
                layer.layer_index + 1,
                GateLayer(layer_index=layer.layer_index + 1,
                        is_odd=layer.is_odd,
                        n_sites=orig.n_sites,
                        gates=[]),
            )
            target.add_gate(
                Gate(
                    matrix=matrix,
                    qubits=gate.qubits,
                    layer_index=target.layer_index,
                    name="Abs",
                    decomposition_part="Abs",
                    original_gate_qubits=gate.qubits,
                )
            )

    # last layer (is independent of other gates)
    last_layer_idx = last_layer.layer_index
    for gate in last_layer.iterate_gates():
        _, middle, post = gate_decomp[id(gate)]
        first, second = gate.qubits
        upper = post[first]
        lower = post[second]
        matrix = jnp.kron(upper, lower) @ middle
        target = new_layers.setdefault(
            last_layer_idx + 1,
            GateLayer(layer_index=last_layer_idx+1,
                    is_odd=layer.is_odd,
                    n_sites=orig.n_sites,
                    gates=[]),
        )
        target.add_gate(
            Gate(
                matrix=matrix,
                qubits=gate.qubits,
                layer_index=target.layer_index,
                name="Abs",
                decomposition_part="Abs",
                original_gate_qubits=gate.qubits,
            )
        )
                     
    # note the list should be long orig.num_layers+1
    layer_list = [lay for lay in new_layers.values()]
    return Circuit(
        n_sites=orig.n_sites,
        dtype=orig.dtype,
        layers=layer_list,
        hamiltonian_type=orig.hamiltonian_type,
        trotter_params=orig.trotter_params,
    )
