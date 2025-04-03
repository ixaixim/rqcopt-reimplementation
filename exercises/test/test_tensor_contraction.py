import numpy as np
from numpy.testing import assert_allclose
import os
import sys
from scipy.linalg import expm
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))
from tensor_contraction import absorb_to_the_right, absorb_to_the_left

class TestContraction(unittest.TestCase):
    def test_absorb_to_the_right(self):
        N = 6
        i = 0
        dim = 2**N

        rng = np.random.default_rng(42)
        # Generate a random two-qubit gate (4x4 complex matrix)
        two_qubit_gate = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        
        # Generate a random big tensor (2^N x 2^N complex matrix)
        big_tensor = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))

        # Compute the result using absorb_two_qubit_gate
        result = absorb_to_the_right(two_qubit_gate, big_tensor, i, N)
        
        # Construct the expected full operator:
        # For qubits 0 and 1: two_qubit_gate (4x4)
        # For qubits 2 and 3: identity (4x4)
        # For qubits 4 and 5: identity (4x4)
        id_2q = np.eye(4)
        full_operator = np.kron(np.kron(two_qubit_gate, id_2q), id_2q)
        
        # Apply the full operator to the big tensor (left multiplication)
        expected = full_operator @ big_tensor
        
        # Assert that result and expected are close within tolerance.
        assert_allclose(result, expected, rtol=1e-6, atol=1e-8)


    def test_absorb_to_the_left(self):
        N = 6
        i = 1
        dim = 2**N

        rng = np.random.default_rng(42)
        # Generate a random two-qubit gate (4x4 complex matrix)
        two_qubit_gate = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
        
        # Generate a random big tensor (2^N x 2^N complex matrix)
        big_tensor = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))

        # Compute the result using absorb_two_qubit_gate
        result = absorb_to_the_left(two_qubit_gate, big_tensor, i, N)
        
        # Construct the expected full operator:
        # For qubits 0: identity (2x2)
        # For qubits and 1 and 2: two_qubit_gate (4x4)
        # For qubits 3,4,5: identity (8x8)
        id_3q = np.eye(8, dtype=complex)
        id_2q = np.eye(4, dtype=complex)
        id_1q = np.eye(2, dtype=complex)

        full_operator = np.kron(np.kron(id_1q, two_qubit_gate), id_3q)
        
        # Apply the full operator to the big tensor (left multiplication)
        expected = big_tensor @ full_operator
        
        # Assert that result and expected are close within tolerance.
        assert_allclose(result, expected, rtol=1e-6, atol=1e-8)

if __name__ == '__main__':
    unittest.main()


