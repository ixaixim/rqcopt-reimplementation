import numpy as np
import unittest
from numpy.testing import assert_allclose

import os
import sys
from scipy.linalg import expm
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))
from mpo import mpo_from_matrix


def reconstruct_matrix_from_mpo(mpo):
    """
    Contract the MPO tensors to reconstruct the original matrix.
    
    Each tensor in the MPO is assumed to have shape (2, 2, bond_left, bond_right),
    where the first two indices are the physical indices (row and column parts).
    The contraction should be done over the bond indices (right index of tensor i
    with left index of tensor i+1).
    
    After full contraction, we will have a tensor of shape:
      (p0, q0, p1, q1, ..., p_{N-1}, q_{N-1}),
    which is then rearranged so that the row physical indices (p's) and column physical
    indices (q's) are grouped, and reshaped into a (2^N x 2^N) matrix.
    """
    # Start with the first tensor.
    T = mpo[0]
    # Sequentially contract over the bond dimension:
    # Contract T's last axis (bond right) with next tensor's third axis (bond left).
    for tensor in mpo[1:]:
        T = np.tensordot(T, tensor, axes=([len(T.shape)-1], [2])) # last axis of first one and axis [2] of second one
    # Suppose there are N MPO tensors.
    N = len(mpo)
    # Reshape into a matrix of shape (2**N, 2**N)
    T = np.squeeze(T)
    # first kets, then bras: reorder axes. 
    new_order = list(range(0, tensor.ndim, 2)) + list(range(1, tensor.ndim, 2))
    T = np.transpose(tensor, axes=new_order)
    return T.reshape(2**N, 2**N)

class TestMPO(unittest.TestCase):
    def test_mpo_reconstruction(self):
        N = 3
        dim = 2 ** N
        # Create a random 2^N x 2^N matrix.
        U_ref = np.random.randn(dim, dim)
        # Convert the matrix into an MPO representation.
        mpo = mpo_from_matrix(U_ref, N)
        # Reassemble the matrix from the MPO tensors.
        U_reconstructed = reconstruct_matrix_from_mpo(mpo)
        # Check that the reassembled matrix is close to the original.
        assert_allclose(U_reconstructed, U_ref, atol=1e-6)

if __name__ == '__main__':
    unittest.main()
