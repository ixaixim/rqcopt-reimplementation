

def compute_gate_environment_tensor(gate_qubits: Tuple[int, ...], E_top_layer: List[jnp.ndarray], E_bottom_layer: List[jnp.ndarray], E_left_boundary: jnp.ndarray, E_right_boundary: jnp.ndarray) -> jnp.ndarray:
    pass

def calculate_cost_function(circuit: Circuit, mpo_ref: MPO, max_bondim: int, **kwargs) -> float:

# TODO: cost function = derivative + plug in the gate