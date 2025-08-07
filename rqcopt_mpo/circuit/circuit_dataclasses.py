import rqcopt_mpo.jax_config

from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional, Any, List, Dict, Set, Iterator
import copy
import jax

# The objects below use the @dataclass decorator for simpler init, repr, (optional immutability, which is a property of jax.numpy). 
# note: field is a way to customize how each attribute is handled by @dataclass decorator. 
#       field(default_factory=...) makes sure that the params tuple is not shared across all Gate instantiations. Avoids shared mutable defaults (here not so necessary since tuple is immutable, but it is good practice)
@dataclass
class Gate:
    """Represents `a single gate acting on one or two` qubits."""
    matrix: np.ndarray            # The numerical matrix (2x2 or 4x4)
    qubits: Tuple[int, ...]       # Tuple of qubit indices it acts on (e.g., (q,) or (q1, q2))
    layer_index: int              # The index of the layer this gate belongs to conceptually
    
    # Optional metadata for clarity and tracking
    name: str = ""                # e.g., "ID", "RX", "CNOT", "K1l", "ExpXYZ", "K2r"
    params: Tuple[Any, ...] = field(default_factory=tuple) # e.g., rotation angle, or (a,b,c) for ExpXYZ 
    
    # --- Metadata specific to decomposed gates ---
    # Link back to the original 2-qubit gate it came from
    original_gate_qubits: Optional[Tuple[int, int]] = None 
    decomposition_part: Optional[str] = None # e.g., "K1l", "K1r", "Exp", "K2l", "K2r"

    # validation method used after initialization
    def __post_init__(self):
        # Basic validation (optional but recommended)
        expected_dim = 2**len(self.qubits)
        if self.matrix.shape != (expected_dim, expected_dim):
            raise ValueError(f"Tensor shape {self.matrix.shape} incompatible with qubits {self.qubits}")

    @property # note: we can cache properties, access is attribute-like. 
    def tensor(self) -> np.ndarray:
        """
        Returns the gate as a 4-index tensor.
        For a single-qubit gate, simply return the tensor.
        For a two-qubit gate stored as 4x4 (out, in), convert it to shape (2,2,2,2) with
        indices ordered as (out_1, out_2, in1, in2).


        Here go the next gates in the circuit............................                                                
                                                  
           │                    out1│     │  out2 
      out  │ (rows)                 │     │       
           │                        │     │       
    ┌──────┼──────┐              ┌──┴─────┴────┐  
    │             │              │             │  
    │             │      ───     │             │  
    │             │              │             │  
    │             │      ───     │             │  
    └──────┬──────┘              └──┬──────┬───┘  
           │                        │      │      
           │                        │      │      
       in  │ (cols)                 │      │      
                                                  
                                  in1      in2    
                                                  
        Here go the previous gates in the circuit.........................
        """
        if self.is_single_qubit():
            return self.matrix  # shape (2,2)
        elif self.is_two_qubit(): # matrix index: (out, in) (or (rows, cols) if you prefer)
                return self.matrix.reshape(2, 2, 2, 2) # (out1, out1, in1, in2) (in tensor network diagram, )
        else:
            raise ValueError("Unsupported gate qubit number.")

    # Add __hash__ and __eq__ for using Gates as keys in dicts or as members of sets, if modifying in place
    def __hash__(self):
        # If tensors could be identical but represent different logical gates, need a more robust hash
        return id(self) # memory address of the object

    def __eq__(self, other):
        if not isinstance(other, Gate):
            return NotImplemented
        return id(self) == id(other) # two Gate instances are only equal if they are literally the same object. 
                
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1

    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2
    
    def copy(self) -> 'Gate': # annotation: the class name Gate has not been fully defined yet. Python will delay the evaluation until after the class is defined. 
        """
        Creates a deep copy of this Gate instance.

        The numerical data in `matrix` is copied.
        Immutable attributes (qubits, layer_index, name, etc.) are assigned directly.
        The `params` tuple is deepcopied to handle potential mutable elements within it.
        """
        # Deep copy the numerical matrix. 
        # Deep copy 'params'. While the tuple itself is immutable, its elements
        # (specified by 'Any') could be mutable. copy.deepcopy ensures these are also copied independently.
        # Tuples of primitive types (e.g. self.qubits), and primitive types (e.g. self.name) are immutable (i.e. types that cannot be changed in place). Assigning an immutable type will effectively copy it. 
        copied_matrix = self.matrix.copy()

        copied_params = copy.deepcopy(self.params)

        new_gate = Gate(
            matrix=copied_matrix,
            qubits=self.qubits,
            layer_index=self.layer_index,
            name=self.name,
            params=copied_params,
            original_gate_qubits=self.original_gate_qubits,
            decomposition_part=self.decomposition_part
        )
        return new_gate


@dataclass
class GateLayer:
    """Represents a single layer in the brickwall circuit."""
    layer_index: int
    is_odd: bool          # True if it's an even layer (acts on (0,1), (2,3)...), False for odd ((1,2), (3,4)...)
    gates: List[Gate] = field(default_factory=list) # All gates conceptually in this layer
    n_sites: Optional[int] = None
    # TODO: add __post_init__ method to check that the gates in layer can be computed in parallel (i.e. no overlapping gates on the same qubits) 

    def iterate_gates(self, reverse: bool = False) -> Iterator[Gate]:
        """
        Provides an iterator over the gates in the layer, sorted spatially.

        This is the recommended way to iterate for MPO/TEBD algorithms.
        The sorting key `g.qubits[0]` correctly orders gates based on their
        "leftmost" qubit index, naturally handling both 1- and 2-qubit gates.

        Args:
            reverse: If False, iterates from left-to-right (ascending qubit index).
                     If True, iterates from right-to-left (descending qubit index).

        Yields:
            Gate objects in the specified spatial order.
        """
        # Sort the gates based on the index of their first qubit.
        # This provides a consistent spatial ordering from left to right.
        sorted_gates = sorted(self.gates, key=lambda g: g.qubits[0], reverse=reverse)
        yield from sorted_gates

    def add_gate(self, gate: Gate):
        if gate.layer_index != self.layer_index:
             # You might want a warning or error here depending on strictness
             print(f"Warning: Adding gate with layer_index {gate.layer_index} to Layer {self.layer_index}")
        self.gates.append(gate)
    
    def copy(self) -> 'GateLayer':
        """
        Creates a deep copy of this GateLayer instance.

        The list of gates is newly created, and each Gate within
        that list is a deep copy of the original gate.
        Primitive attributes (layer_index, is_odd) are copied by value.
        """
        copied_gates = [gate.copy() for gate in self.gates]

        new_layer = GateLayer(
            layer_index=self.layer_index,
            is_odd=self.is_odd,
            gates=copied_gates
        )
        return new_layer



@dataclass
class Circuit:
    """Represents the entire quantum circuit structure."""
    n_sites: int
    dtype: Any = jnp.complex128    # Optional: Add original circuit generation parameters if needed
    layers: List[GateLayer] = field(default_factory=list)
    
    hamiltonian_type: Optional[str] = None
    trotter_params: Optional[dict] = None # e.g., {'t': 0.1, 'n_repetitions': 2, ...}

    @property
    def num_layers(self):
        return len(self.layers)
    
    @property
    def num_gates(self):
        n = 0
        for layer in self.layers:
            for gate in layer.gates:
                n += 1
        return n
    
    def __post_init__(self):
        """ validation after initialization."""
        # You could even add a check to ensure all gates match the circuit's dtype.
        for layer in self.layers:
            for gate in layer.gates:
                if gate.matrix.dtype != self.dtype:
                    print(f"Warning: Gate {gate.name} on qubits {gate.qubits} has dtype {gate.matrix.dtype}, "
                          f"which differs from Circuit's dtype {self.dtype}.")

    
    def sort_layers(self):
        """Ensures layers are sorted by index."""
        self.layers.sort(key=lambda layer: layer.layer_index)

    def print_gates(self, max_per_layer=10):
         print(f"Circuit with {self.n_sites} sites and {len(self.layers)} layers.")
         if not self.layers:
             print("Circuit is empty.")
             return
         for layer in self.layers:
            print(f"--- Layer {layer.layer_index} (Orig Odd: {layer.is_odd}) --- Gates: {len(layer.gates)}")
            count = 0
            for gate in layer.gates:
                if count < max_per_layer:
                    # Ensure tensor is on CPU for printing shape if using JAX GPU
                    shape_str = str(np.shape(gate.matrix)) 
                    print(f"  {gate.name} ({gate.decomposition_part}) on {gate.qubits} shape:{shape_str}")
                elif count == max_per_layer:
                    print(f"  ... (omitting remaining {len(layer.gates) - count} gates)")
                count += 1
            if not layer.gates:
                print("  (Layer is empty)")

    def copy(self) -> 'Circuit':
        """Creates a deep copy of this Circuit instance."""
        # Deep copy layers: iterate and call GateLayer's copy method
        copied_layers = [layer.copy() for layer in self.layers]

        # Deep copy trotter_params if it exists (dictionaries are mutable)
        copied_trotter_params = copy.deepcopy(self.trotter_params) if self.trotter_params is not None else None

        new_circuit = Circuit(
            n_sites=self.n_sites, # int is immutable
            dtype=self.dtype,
            layers=copied_layers,
            hamiltonian_type=self.hamiltonian_type, # str is immutable
            trotter_params=copied_trotter_params
        )
        return new_circuit
    
    def to_matrix(self) -> jnp.ndarray:
        self.sort_layers()
        total_circuit_matrix = jnp.eye(2**self.n_sites, dtype=self.dtype)
        
        for layer in reversed(self.layers):

            sorted_gates_in_layer = sorted(layer.gates, key=lambda g: g.qubits[0])

            current_q_idx = 0
            gate_list_iter_idx = 0
            ops_for_kron_product = []

            while current_q_idx < self.n_sites:
                is_identity_for_site = True
                if gate_list_iter_idx < len(sorted_gates_in_layer):
                    gate = sorted_gates_in_layer[gate_list_iter_idx]
                    if gate.qubits[0] == current_q_idx: 
                        ops_for_kron_product.append(jnp.array((gate.matrix), dtype=self.dtype))
                        current_q_idx += len(gate.qubits) # Advance by the number of qubits in the gate
                        gate_list_iter_idx += 1
                        is_identity_for_site = False

                if is_identity_for_site:
                    ops_for_kron_product.append(jnp.eye(2, dtype=self.dtype)) # Identity for this site
                    current_q_idx += 1

            # Build the full layer matrix from the collected ops
            layer_op_matrix_k = ops_for_kron_product[0]
            for op_idx in range(1, len(ops_for_kron_product)):
                layer_op_matrix_k = jnp.kron(layer_op_matrix_k, ops_for_kron_product[op_idx])

            # total_circuit_matrix was called next_layer_matrix in your code
            total_circuit_matrix = total_circuit_matrix @ layer_op_matrix_k 
        
        return total_circuit_matrix
                        



    # --- You would add methods for compression, analysis, simulation here ---
    # def compress(self, ...)
    # def apply_to_state(self, ...)
    # def contract_layers(self, ...)

