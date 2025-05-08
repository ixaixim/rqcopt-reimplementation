from dataclasses import dataclass, field
import numpy as np
from typing import Tuple, Optional, Any, List, Dict, Set

# The objects below use the @dataclass decorator for simpler init, repr, (optional immutability, which is a property of jax.numpy). 
# note: field is a way to customize how each attribute is handled by @dataclass decorator. 
#       field(default_factory=...) makes sure that the params tuple is not shared across all Gate instantiations. 
@dataclass
class Gate:
    """Represents a single gate acting on one or two qubits."""
    tensor: np.ndarray            # The numerical tensor (2x2 or 4x4)
    qubits: Tuple[int, ...]       # Tuple of qubit indices it acts on (e.g., (q,) or (q1, q2))
    layer_index: int              # The index of the layer this gate belongs to conceptually
    
    # Optional metadata for clarity and tracking
    name: str = ""                # e.g., "ID", "RX", "CNOT", "K1l", "ExpXYZ", "K2r"
    params: Tuple[Any, ...] = field(default_factory=tuple) # e.g., rotation angle, or (a,b,c) for ExpXYZ
    
    # --- Metadata specific to decomposed gates ---
    # Link back to the original 2-qubit gate it came from
    original_gate_qubits: Optional[Tuple[int, int]] = None 
    decomposition_part: Optional[str] = None # e.g., "K1l", "K1r", "Exp", "K2l", "K2r"

    @property # note: we can cache properties, access is attribute-like. 
    def tensor_4d(self) -> np.ndarray:
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
            return self.tensor  # shape (2,2)
        elif self.is_two_qubit():
            if self.tensor.shape == (4, 4): # index: (out, in) (or (rows, cols) if you prefer)
                tensor_4d = self.tensor.reshape(2, 2, 2, 2) # (out1, out1, in1, in2) (in tensor network diagram, )
                return tensor_4d #np.transpose(tensor_4d, (2, 3, 0, 1)) 
            elif self.tensor.shape == (2, 2, 2, 2):
                return self.tensor
            else:
                raise ValueError(f"Unexpected two-qubit gate tensor shape: {self.tensor.shape}")
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
    
    # explanation: we need to distinguish gates based on their role in the circuit, not in the data. 
    # # Create two Gate objects with identical tensors and other parameters,
    # # but they represent different logical operations in the circuit.
    # gate1 = Gate(tensor=identity_tensor, qubits=(0,), layer_index=0, name="I_gate")
    # gate2 = Gate(tensor=identity_tensor, qubits=(1,), layer_index=1, name="I_gate")

    # # Even though gate1.tensor and gate2.tensor are identical,
    # # gate1 and gate2 are distinct objects.
    # print(gate1 == gate2)  # This will print: False


    # validation method used after initialization
    def __post_init__(self):
        # Basic validation (optional but recommended)
        expected_dim = 2**len(self.qubits)
        if self.tensor.shape != (expected_dim, expected_dim):
            raise ValueError(f"Tensor shape {self.tensor.shape} incompatible with qubits {self.qubits}")
            
    def is_single_qubit(self) -> bool:
        return len(self.qubits) == 1

    def is_two_qubit(self) -> bool:
        return len(self.qubits) == 2

@dataclass
class GateLayer:
    """Represents a single layer in the brickwall circuit."""
    layer_index: int
    is_odd: bool          # True if it's an odd layer (acts on (0,1), (2,3)...), False for even ((1,2), (3,4)...)
    gates: List[Gate] = field(default_factory=list) # All gates conceptually in this layer

    def add_gate(self, gate: Gate):
        if gate.layer_index != self.layer_index:
             # You might want a warning or error here depending on strictness
             print(f"Warning: Adding gate with layer_index {gate.layer_index} to Layer {self.layer_index}")
        self.gates.append(gate)

@dataclass
class Circuit:
    """Represents the entire quantum circuit structure."""
    n_sites: int
    layers: List[GateLayer] = field(default_factory=list)
    
    # Optional: Add original circuit generation parameters if needed
    hamiltonian_type: Optional[str] = None
    trotter_params: Optional[dict] = None # e.g., {'t': 0.1, 'n_repetitions': 2, ...}

    def sort_layers(self):
        """Ensures layers are sorted by index."""
        self.layers.sort(key=lambda layer: layer.layer_index)

    # def get_layer(self, index: int) -> Optional[GateLayer]:
    #     """Finds a layer by its index."""
    #     for layer in self.layers:
    #         if layer.layer_index == index:
    #             return layer
    #     return None
        
    # def get_all_gates_flat(self) -> List[Gate]:
    #     """Returns a flat list of all gates, sorted by layer."""
    #     self.sort_layers()
    #     all_gates = []
    #     for layer in self.layers:
    #         all_gates.extend(layer.gates)
    #     return all_gates

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
                    shape_str = str(np.shape(gate.tensor)) 
                    print(f"  {gate.name} ({gate.decomposition_part}) on {gate.qubits} shape:{shape_str}")
                elif count == max_per_layer:
                    print(f"  ... (omitting remaining {len(layer.gates) - count} gates)")
                count += 1
            if not layer.gates:
                print("  (Layer is empty)")

    def absorb_single_qubit_gates(self, max_passes: int = 1):
        # TODO: this is a general technique to absorb. It works. However, need a better one for just our trotterized case. 
        # TODO: consider a merge_layer function instead. It will be used inside absorb_single_qubit_gates. 
        """
        Iteratively absorbs consecutive single-qubit gates acting on the same qubit.

        Compares adjacent layers (L_i, L_{i+1}). If a single-qubit gate on
        qubit q exists in both, they are multiplied (gate_{i+1} @ gate_i)
        and replaced by a single gate placed in layer L_i. The original
        gates are removed. This process repeats until no more absorptions
        can be made in a full pass, or max_passes is reached.
        Empty layers are removed, and layers are re-indexed at the end.

        Args:
            max_passes: Maximum number of sweeps through the layers to absorb gates.
        """
        if not self.layers or len(self.layers) < 2:
            print("Absorption skipped: Not enough layers.")
            return 0 # No absorptions possible

        total_absorbed_count = 0
        for pass_num in range(max_passes):
            print(f"\n--- Absorption Pass {pass_num + 1} ---")
            absorptions_in_pass = 0
            
            # Keep track of gates to remove/add for this pass
            # Use gate IDs as keys in remove_map for safety if gates are somehow duplicated
            gates_to_remove_map: Dict[int, Set[int]] = {layer.layer_index: set() for layer in self.layers}
            gates_to_add_map: Dict[int, List[Gate]] = {layer.layer_index: [] for layer in self.layers}
            
            # Need to iterate using indices because the list length might change conceptually
            # But act on copies or use the map approach to avoid modification issues
            current_layers = self.layers # Operate on the list from the previous pass or start

            for i in range(len(current_layers) - 1):
                layer_curr = current_layers[i]
                layer_next = current_layers[i+1]
                
                idx_curr = layer_curr.layer_index
                idx_next = layer_next.layer_index

                # Create a lookup for single-qubit gates in the *next* layer by qubit index
                # Important: Only consider gates NOT already marked for removal in this pass
                next_layer_sq_gates: Dict[int, Gate] = {}
                remove_set_next = gates_to_remove_map.get(idx_next, set())
                for gate_next in layer_next.gates:
                     # Check if single qubit AND not already marked for removal
                    if gate_next.is_single_qubit() and id(gate_next) not in remove_set_next:
                        q_next = gate_next.qubits[0]
                        if q_next in next_layer_sq_gates:
                           # This shouldn't happen if layers truly represent parallel ops, but good to check
                           print(f"Warning: Multiple single-qubit gates found on qubit {q_next} in layer {idx_next}. Using the first one found.")
                        else:
                           next_layer_sq_gates[q_next] = gate_next

                # Iterate through gates in the current layer
                remove_set_curr = gates_to_remove_map.get(idx_curr, set())
                for gate_curr in layer_curr.gates:
                    # Check if single qubit AND not already marked for removal
                    if not gate_curr.is_single_qubit() or id(gate_curr) in remove_set_curr:
                        continue

                    q_curr = gate_curr.qubits[0]

                    # Find a partner in the next layer's lookup
                    gate_next = next_layer_sq_gates.get(q_curr)

                    if gate_next:
                        # --- Found a pair to absorb ---
                        absorptions_in_pass += 1
                        
                        # Perform absorption: New tensor = Next Gate * Current Gate
                        # Ensure tensors are compatible (e.g., both (2,2))
                        if gate_curr.tensor.shape == (2,2) and gate_next.tensor.shape == (2,2):
                             new_tensor = gate_next.tensor @ gate_curr.tensor
                        else:
                             print(f"Warning: Skipping absorption on qubit {q_curr} between layers {idx_curr} and {idx_next} due to incompatible shapes: {gate_curr.tensor.shape} and {gate_next.tensor.shape}")
                             continue # Skip this absorption

                        # Create the new absorbed gate - place it in the current layer (layer i)
                        new_gate = Gate(
                            tensor=new_tensor,
                            qubits=(q_curr,),
                            layer_index=idx_curr, # Belongs to the layer of the first gate
                            name="AbsorbedSQ",
                            params=(),
                            original_gate_qubits=None, # Origin is mixed
                            decomposition_part="Absorbed"
                        )

                        # Mark original gates for removal (use IDs for safety)
                        gates_to_remove_map.setdefault(idx_curr, set()).add(id(gate_curr))
                        gates_to_remove_map.setdefault(idx_next, set()).add(id(gate_next))

                        # Add the new gate to the current layer's addition list
                        gates_to_add_map.setdefault(idx_curr, []).append(new_gate)

                        # Remove the consumed 'next' gate from lookup to prevent re-use in this pass
                        del next_layer_sq_gates[q_curr]

            # --- Update the circuit layers for the next pass ---
            if absorptions_in_pass == 0:
                 print("No more absorptions found in this pass.")
                 break # Exit loop if no changes were made

            new_layers_intermediate = []
            for layer in current_layers:
                idx = layer.layer_index
                current_gates = layer.gates
                remove_ids = gates_to_remove_map.get(idx, set())
                add_list = gates_to_add_map.get(idx, [])

                # Filter out removed gates and add new ones
                final_gates = [g for g in current_gates if id(g) not in remove_ids] + add_list

                # Only keep layer if it has gates
                if final_gates:
                     # Create a new layer object. Keep original 'is_odd' for context.
                     # Layer index will be fixed later.
                    new_layers_intermediate.append(GateLayer(
                        layer_index=-1, # Temporary index
                        is_odd=layer.is_odd,
                        gates=final_gates
                    ))
            
            # Update self.layers for the next iteration or final result
            self.layers = new_layers_intermediate 
            total_absorbed_count += absorptions_in_pass
            print(f"Absorbed {absorptions_in_pass} gate pairs in pass {pass_num + 1}. Total layers now: {len(self.layers)}")

        # --- Final Cleanup: Remove empty layers and re-index ---
        print("\n--- Finalizing Absorption ---")
        final_layers = [layer for layer in self.layers if layer.gates] # Filter empty
        
        # Re-index the remaining layers sequentially
        for i, layer in enumerate(final_layers):
            layer.layer_index = i # Assign new sequential index
            # Also update layer_index for all gates within the layer
            for gate in layer.gates:
                gate.layer_index = i

        self.layers = final_layers
        print(f"Absorption complete. Total pairs absorbed: {total_absorbed_count}. Final layer count: {len(self.layers)}")
        
        # self.print_gates() # Optional: print final structure

        return total_absorbed_count

    # --- You would add methods for compression, analysis, simulation here ---
    # def compress(self, ...)
    # def apply_to_state(self, ...)
    # def contract_layers(self, ...)

