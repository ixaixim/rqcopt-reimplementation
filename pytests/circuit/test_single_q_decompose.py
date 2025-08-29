import rqcopt_mpo.jax_config
import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Assuming the following functions and classes are defined and accessible as per the problem description.
from rqcopt_mpo.circuit.circuit_dataclasses import Gate, GateLayer, Circuit
from rqcopt_mpo.circuit.circuit_builder import _random_unitary
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import weyl_decompose_circuit
from rqcopt_mpo.circuit.decompose.single_q_decompose import euler_zyz_decompose_circuit, zyz_decompose_gate
from scipy.linalg import polar


def make_unitary_128(M) -> np.ndarray:
    """
    Ensure M is a NumPy complex128, unitary matrix via polar decomposition.
    Returns the unitary factor U from M = U H.
    """
    M64 = np.asarray(M, dtype=np.complex128)
    U, _ = polar(M64)  # U is unitary, _ is Hermitian PSD
    return U

@pytest.mark.parametrize("seed", [0, 1, 42])
def test_decomposition_preserves_unitary(seed):
    """
    Tests that the full circuit decomposition (Weyl + ZYZ) preserves the
    overall unitary matrix of the circuit.
    """
    # ARRANGE: Set up the initial circuit from random unitary gates.
    key = jax.random.PRNGKey(seed)
    k0, k1, k2 = jax.random.split(key, 3)

    # # Make three 2-qubit unitaries (4x4)
    U01 = _random_unitary(4, key=k0, dtype=jnp.complex64)
    U23 = _random_unitary(4, key=k1, dtype=jnp.complex64)
    U12 = _random_unitary(4, key=k2, dtype=jnp.complex64)

    # # Ensure they are perfectly unitary and have the correct dtype for the circuit
    U01 = make_unitary_128(U01)
    U23 = make_unitary_128(U23)
    U12 = make_unitary_128(U12)

    # Create Gate instances
    g01 = Gate(matrix=U01, qubits=(0, 1), layer_index=0, name="U2")
    g23 = Gate(matrix=U23, qubits=(2, 3), layer_index=0, name="U2")
    g12 = Gate(matrix=U12, qubits=(1, 2), layer_index=1, name="U2")

    # # Assemble the layers and the final circuit
    # layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g01, g23])
    # layer1 = GateLayer(layer_index=1, is_odd=True, gates=[g12])
    # original_circuit = Circuit(n_sites=4, dtype=jnp.complex128, layers=[layer0, layer1])

    # debug: multiple single qubit gates
    # U0 = _random_unitary(2, key=k0, dtype=jnp.complex128)
    # U0 = make_unitary_128(U0)
    # g0 = Gate(matrix=U0, qubits=(3,), layer_index=0, name="U0")
    # U1 = _random_unitary(2, key=k1, dtype=jnp.complex128)
    # U1 = make_unitary_128(U1)
    # g1 = Gate(matrix=U1, qubits=(3,), layer_index=0, name="U1")
    # U3 = _random_unitary(2, key=k2, dtype=jnp.complex128)
    # U3 = make_unitary_128(U3)
    # g3 = Gate(matrix=U3, qubits=(3,), layer_index=0, name="U3")

    # print(f"g0: \n{g0.matrix}")
    # print(f"g1: \n{g1.matrix}")
    # print(f"g3: \n{g3.matrix}")
    # layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g3]) 
    layer0 = GateLayer(layer_index=0, is_odd=False, gates=[g01, g23]) 
    layer1 = GateLayer(layer_index=1, is_odd=False, gates=[g12]) 
    original_circuit = Circuit(n_sites=4, dtype=jnp.complex128, layers=[layer0, layer1])
    # Calculate the unitary of the original, undecomposed circuit.
    # This requires a 'to_matrix()' method on the Circuit object.
    original_matrix = original_circuit.to_matrix()

    # ACT: Perform the two-step decomposition.
    weyl_decomposed_circuit = weyl_decompose_circuit(original_circuit)
    # weyl_decomposed_circuit.print_gates()
    # TODO: check that decomposed zyz circuit is the same  as original_circuit (so far, only weyl_decomposed_circuit satisfies the equality.)
    final_decomposed_circuit = euler_zyz_decompose_circuit(weyl_decomposed_circuit, include_global_phase=False)
    final_decomposed_circuit.print_gates()
    
    # Calculate the unitary of the final, fully decomposed circuit.
    decomposed_matrix = final_decomposed_circuit.to_matrix()
    phase_align_decomposed_matrix = phase_align(original_matrix, decomposed_matrix)
    # ASSERT: Check that the matrix before and after decomposition is the same.
    # We use np.allclose to account for floating point precision issues.

    # debug:
    A = original_matrix
    B = phase_align_decomposed_matrix
    mask = A != B
    diff_indices = np.argwhere(mask)
    for i, j in diff_indices:
        print(f"At ({i},{j}): A={A[i,j]} vs B={B[i,j]}")

    assert A.shape == B.shape, "Matrices must have the same shape."
    assert np.allclose(A, B), "The unitary matrix should not change after decomposition."


# utilities to test single qubit Gate dataclass decomposition
def random_unitary_2x2(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(X)
    # Normalize to unitary (fix global phase from R's diag)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q @ np.diag(ph)

def reconstruct(parts):
    """Multiply gate matrices in ascending layer_index order."""
    parts_sorted = sorted(parts, key=lambda g: g.layer_index)
    M = np.eye(2, dtype=np.complex128)
    for g in parts_sorted:
        M = g.matrix @ M
    return M

def phase_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return B multiplied by the optimal global phase to align with A."""
    # e^{i phi} that maximizes Re Tr(A^\dagger e^{i phi} B) -> phi = -arg Tr(A^\dagger B)
    num = np.vdot(A, B)  # Tr(A^\dagger B) for 2x2 using Frobenius inner product
    if np.abs(num) < 1e-16:
        return B
    return np.exp(-1j * np.angle(num)) * B

@pytest.mark.parametrize("seed,out_layer", [(0, 7), (1, 3), (42, 10)])
@pytest.mark.parametrize("include_global_phase", [True, False])
def test_zyz_reconstruction(seed, out_layer, include_global_phase):
    # verifies the ZYZ reconstrunction equals the original 1-qubit gate
    #   (exactly when include_global_phase=True, and up to a global phase otherwise).
    U = random_unitary_2x2(seed)

    # minimal Gate instance (adjust args if your dataclass differs)
    g = Gate(
        matrix=U,
        qubits=(0,),
        layer_index=0,
        name="U",
        decomposition_part=None,
        params=(),
        original_gate_qubits=(0,),
    )

    phase, parts = zyz_decompose_gate(g, out_layer)

    # Basic sanity on returned parts
    assert all(p.qubits == (0,) for p in parts)
    # Layers should be consecutive starting at out_layer (optionally with a phase gate)
    layer_indices = sorted(p.layer_index for p in parts)
    expect_len = 3
    assert len(parts) == expect_len
    assert layer_indices == list(range(out_layer, out_layer + expect_len))

    # Reconstruction
    W = reconstruct(parts)

    if include_global_phase:
        # exact equality (up to numerical tolerance), since a phase gate is explicitly included
        np.testing.assert_allclose(W*np.exp(1j*phase), U, rtol=0, atol=1e-12)
    else:
        # equality up to a global phase
        W_aligned = phase_align(U, W)
        np.testing.assert_allclose(W_aligned, U, rtol=0, atol=1e-12)


