import os
import sys
import unittest
import numpy as np

# Adjust the path so that we can import the modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))

from tensor_gradient import environment_tensor_ABC, environment_tensor_simple_brickwall
from brickwall_circuit import brickwall_circuit
from tensor_contraction import absorb_to_the_right, absorb_to_the_left

class TestTraceGradient(unittest.TestCase):

    def test_environment_tensor_ABC(self):
        # Create random matrices A (5x3) and C (4x5)
        A = np.random.randn(5, 3)
        C = np.random.randn(4, 5)
        
        # Compute the gradient using the environment_tensor_ABC function.
        G = environment_tensor_ABC(A, C)  # Expected shape is (3, 4)
        
        # For each entry (l, m) in G, verify that the partial derivative equals np.trace(A@B@C)
        for l in range(3):
            for m in range(4):
                # Create a test matrix B with a single 1 at (l, m)
                B = np.zeros((3, 4))
                B[l, m] = 1
                # For this B, the trace computes to:
                trace_value = np.trace(A @ B @ C)
                # Assert that G[l, m] is close to the computed trace value.
                self.assertAlmostEqual(G[l, m], trace_value, places=5)

    def trace_simple_brickwall(self, even_1, odd_1, even_2, ref):
        # right env
        ref = np.einsum("abnm, nmcdefgh->abcdefgh", even_2[0], ref) #contract two-qubit gate to the right
        ref = np.einsum("cdnm, abnmefgh -> abcdefgh", even_2[1], ref) 
        # left env
        ref = np.einsum("nmef, abcdnmgh -> abcdefgh", even_1[0], ref)
        ref = np.einsum("nmgh, abcdefnm -> abcdefgh", even_1[1], ref)
        # center 
        ref = np.einsum("bcnm, anmdefgh -> abcdefgh", odd_1, ref)
        trace = np.einsum('ijklijkl->', ref)
        return trace



    def test_finite_difference_simple_brickwall(self):
        # Set this flag to True for complex tensors, False for real tensors.
        complex_flag = True

        if complex_flag:
            even_1 = [np.random.randn(2,2,2,2) + 1j * np.random.randn(2,2,2,2),
                      np.random.randn(2,2,2,2) + 1j * np.random.randn(2,2,2,2)]
            odd_1 = np.random.randn(2,2,2,2) + 1j * np.random.randn(2,2,2,2)
            even_2 = [np.random.randn(2,2,2,2) + 1j * np.random.randn(2,2,2,2),
                      np.random.randn(2,2,2,2) + 1j * np.random.randn(2,2,2,2)]
            ref = np.random.randn(*([2]*8)) + 1j * np.random.randn(*([2]*8))
            Delta = np.random.randn(*odd_1.shape) + 1j * np.random.randn(*odd_1.shape)
        else:
            even_1 = [np.random.randn(2,2,2,2), np.random.randn(2,2,2,2)]
            odd_1 = np.random.randn(2,2,2,2)
            even_2 = [np.random.randn(2,2,2,2), np.random.randn(2,2,2,2)]
            ref = np.random.randn(*([2]*8))
            Delta = np.random.randn(*odd_1.shape)

        layer = 1
        pos = 0
        epsilon = 1e-6

        # Finite difference approximation.
        increment = self.trace_simple_brickwall(even_1, odd_1 + (epsilon * Delta), even_2, ref)
        decrement = self.trace_simple_brickwall(even_1, odd_1 - (epsilon * Delta), even_2, ref)
        finite_diff = (increment - decrement) / (2*epsilon)

        # Compute the gradient inner product.
        G = environment_tensor_simple_brickwall(even_1, odd_1, even_2, ref, layer, pos)
        # note: the contraction "absorbs" the conjugation operation. In other words, conjugation is not needed. 
        inner_product = np.einsum('ijkl, klij->', G, Delta)

        # For complex numbers, using np.testing.assert_allclose ensures that both real and imaginary parts are close.
        np.testing.assert_allclose(inner_product, finite_diff, atol=1e-5)


    def test_finite_difference_ABC(self):
        # Flag to switch between real and complex tensors.
        complex_flag = True  # Set to False for real tensors

        epsilon = 1e-6
        if complex_flag:
            # Generate complex matrices by adding random imaginary parts.
            A = np.random.randn(5, 3) + 1j * np.random.randn(5, 3)
            B = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
            C = np.random.randn(4, 5) + 1j * np.random.randn(4, 5)
            Delta = np.random.randn(3, 4) + 1j * np.random.randn(3, 4)
        else:
            A = np.random.randn(5, 3)
            B = np.random.randn(3, 4)
            C = np.random.randn(4, 5)
            Delta = np.random.randn(3, 4)
        
        # Compute the finite difference quotient:
        # (trace(A @ (B+epsilon*Delta) @ C) - trace(A @ B @ C)) / epsilon
        trace_with_perturbation = np.trace(A @ (B + epsilon * Delta) @ C)
        trace_without_perturbation = np.trace(A @ B @ C)
        finite_diff = (trace_with_perturbation - trace_without_perturbation) / epsilon

        # Compute the gradient G and its Frobenius inner product with Delta.
        G = environment_tensor_ABC(A, C)
        inner_product = np.sum(G * Delta)

        # Verify that the finite difference approximates the inner product.
        if complex_flag:
            # Check separately for real and imaginary parts.
            self.assertAlmostEqual(finite_diff.real, inner_product.real, places=5)
            self.assertAlmostEqual(finite_diff.imag, inner_product.imag, places=5)
        else:
            self.assertAlmostEqual(finite_diff, inner_product, places=5)

if __name__ == '__main__':
    unittest.main()
