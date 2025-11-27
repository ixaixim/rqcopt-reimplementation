import rqcopt_mpo.jax_config
import numpy as np
import jax.numpy as jnp

from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit


def test_circuit_roundtrip_json(tmp_path):
    """Circuits survive save/load roundtrip with gates and metadata intact."""
    # simple 1q + 2q gates to exercise both code paths
    x_gate = Gate(
        matrix=jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128),
        qubits=(0,),
        layer_index=0,
        name="X",
        params=("theta",),
        params_dict={"example": 1},
    )
    cnot = Gate(
        matrix=np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=jnp.complex128,
        ),
        qubits=(1, 2),
        layer_index=1,
        name="CNOT",
        original_gate_qubits=(1, 2),
        decomposition_part="control",
    )
    layer0 = GateLayer(layer_index=0, is_odd=False, gates=[x_gate])
    layer1 = GateLayer(layer_index=1, is_odd=True, gates=[cnot], n_sites=3)
    original = Circuit(
        n_sites=3,
        dtype=jnp.complex128,
        layers=[layer0, layer1],
        hamiltonian_type="test",
        trotter_params={"steps": 2},
    )

    path = tmp_path / "circuit.json"
    original.save_json(path)
    restored = Circuit.load_json(path)

    assert restored.n_sites == original.n_sites
    assert np.dtype(restored.dtype) == np.dtype(original.dtype)
    assert restored.hamiltonian_type == "test"
    assert restored.trotter_params == {"steps": 2}
    assert restored.num_layers == 2
    assert restored.num_gates == 2

    np.testing.assert_allclose(
        np.asarray(restored.to_matrix(), dtype=np.complex128),
        np.asarray(original.to_matrix(), dtype=np.complex128),
    )
