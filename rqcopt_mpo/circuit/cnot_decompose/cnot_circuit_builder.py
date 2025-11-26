import rqcopt_mpo.jax_config

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace
from collections import defaultdict


from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
import pennylane as qml

from rqcopt_mpo.circuit.circuit_dataclasses import Circuit, Gate, GateLayer

# Two-qubit basis decomposition that expands every gate into single-qubit
# rotations and CNOTs. The decomposer is global to avoid re-instantiation.
basis = CXGate()
decomposer = TwoQubitBasisDecomposer(basis)

def _pl_ops_to_instructions(ops, dtype: np.dtype):
    """
    Convert a list of PennyLane operations into the lightweight "instruction" format
    the rest of this module expects (operation.num_qubits, matrix, qubits with _index).
    Wire indices are kept as-is (q0 -> 0, q1 -> 1); matrices are computed in a fixed
    global wire order [0, 1] to avoid per-op ordering ambiguity.
    """
    instructions = []
    for op in ops:
        # ignore global phase terms (no wires)
        if len(op.wires) == 0:
            continue

        wire_order = list(op.wires)
        num_qubits = len(wire_order)

        if num_qubits == 1:
            matrix = np.array(qml.matrix(op, wire_order=wire_order), dtype=dtype)
        elif num_qubits == 2:
            # Always evaluate in the global [0,1] order to keep a consistent basis.
            matrix = np.array(qml.matrix(op, wire_order=[0, 1]), dtype=dtype)
        else:
            raise ValueError(f"Unsupported qubit count {num_qubits} for PL conversion.")

        qubits = tuple(SimpleNamespace(_index=int(w)) for w in wire_order)

        instructions.append(
            SimpleNamespace(
                operation=SimpleNamespace(num_qubits=num_qubits),
                matrix=matrix,
                qubits=qubits,
                name=op.name,
                parameters=op.parameters# this is for pennylane use only
            )
        )
    return instructions

def _embed_single_into_two(mat: np.ndarray, position: int, dtype, little_endian: bool) -> np.ndarray:
    """
    Embed a single-qubit matrix on qubit 0 or 1.
    - little_endian=True: qubit 0 is the least-significant bit (Qiskit convention) → I ⊗ U
    - little_endian=False: qubit 0 is the most-significant bit (PennyLane path here) → U ⊗ I
    """
    eye = np.eye(2, dtype=dtype)
    if little_endian:
        if position == 0:
            return np.kron(eye, mat)
        if position == 1:
            return np.kron(mat, eye)
    else:
        if position == 0:
            return np.kron(mat, eye)
        if position == 1:
            return np.kron(eye, mat)
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

def _build_two_qubit_middle_block(
    ops, dtype: np.dtype, from_pennylane=False,
) -> np.ndarray:
    """

    Collapse a sequence containing CNOTs and 1q rotations into a single 4x4 matrix.
    If decomposition is canonical from Pennylane, returns also the parameters of Rz, Ry1, Ry2.
    """
    # TODO: will later also save the parameters of this block (i.e. how many CNOT are present)
    block = np.eye(4, dtype=dtype)
    params = defaultdict(list)
    for instr in ops:
        op = instr.operation
        step = instr.matrix
        if op.num_qubits == 1:
            pos = instr.qubits[0]._index
            step = _embed_single_into_two(step, pos, dtype, little_endian=not from_pennylane)
            if from_pennylane:
                params[instr.name].append(instr.parameters[0])
        elif op.num_qubits != 2:
            raise ValueError("Instruction acting on more than two qubits.")
        block = step @ block

    if from_pennylane:
        return block, params
    else: return block


def _decompose_gate_into_cnot_blocks(
    gate: Gate, basis_fidelity: Optional[float]
) -> Tuple[Dict[int, np.ndarray], np.ndarray, Dict[int, np.ndarray]]:
    """Return (pre, middle, post) blocks for a two-qubit gate."""
    if not gate.is_two_qubit():
        raise ValueError("CNOT decomposition expects a two-qubit gate.")

    dtype = gate.matrix.dtype
    
    use_pennylane = True#basis_fidelity is None

    # if we require decomposition into three CNOT, use the pennylane canonical decomposition.
    if use_pennylane:
        # replacing qiskit QuantumCircuit with pennylane
        ops = qml.ops.two_qubit_decomposition(
            np.array(gate.matrix, dtype=np.complex128), wires=[0, 1]
        )
        instructions = _pl_ops_to_instructions(ops, dtype)
        if len(instructions) < 10:
            raise ValueError(
                f"PennyLane decomposition produced {len(instructions)} instructions; expected at least 10 (excluding global phase)."
            )

        pre_ops = instructions[0:2]
        middle_ops = instructions[2:8]
        post_ops = instructions[8:10]

        pre = _single_ops(pre_ops, dtype)
        middle, params = _build_two_qubit_middle_block(middle_ops, dtype, from_pennylane=use_pennylane)
        post = _single_ops(post_ops, dtype)

        # save the rot gate into gate params dict
        gate.params_dict = params
        return (
            {gate.qubits[pos]: pre[pos] for pos in range(2)},
            middle,
            {gate.qubits[pos]: post[pos] for pos in range(2)},
        )

    else: 
        circ = decomposer(
            Operator(np.array(gate.matrix, dtype=np.complex128)),
            basis_fidelity=basis_fidelity,
        )
        instructions = list(circ.data) # keep the old instructions for reference

        # code written from qiskit:
        cnot_indices = [
            idx for idx, instr in enumerate(instructions) if instr.operation.num_qubits == 2
        ]
        if len(cnot_indices)<3:
            raise ValueError("Current support for decomposition into 3 CNOTs.")
        if not cnot_indices:
            raise ValueError("Basis decomposition did not produce any two-qubit gate.")
        first, last = cnot_indices[0], cnot_indices[-1]
        # TODO: add a check for printing: "gate on q: {gate.qubits} has {len(cnot_indices)} cnots."

        pre_ops = instructions[:first]
        middle_ops = instructions[first : last + 1]
        post_ops = instructions[last + 1 :]

        pre = _single_ops(pre_ops, dtype)
        middle = _build_two_qubit_middle_block(middle_ops, dtype, from_pennylane=use_pennylane)
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
            abs_meta = { 
                "upper": upper,
                "lower": lower,
                **dict(gate.params_dict) # copy and include any existing metadata
            }

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
                    name="CNOT_block",
                    decomposition_part="CNOT_block",
                    original_gate_qubits=gate.qubits,
                    params_dict=abs_meta,
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
            abs_meta = { # TODo: append them
                "upper": upper,
                "lower": lower,
                **dict(gate.params_dict) # copy and include any existing metadata
            }

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
                    name="CNOT_block",
                    decomposition_part="CNOT_block",
                    original_gate_qubits=gate.qubits,
                    params_dict=abs_meta,
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
        abs_meta = { # TODo: append them
            "upper": upper,
            "lower": lower,
            **dict(gate.params_dict) # copy and include any existing metadata
        }

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
                name="CNOT_block",
                decomposition_part="CNOT_block",
                original_gate_qubits=gate.qubits,
                params_dict=abs_meta,
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
