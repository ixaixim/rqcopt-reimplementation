import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../.')))



def environment_tensor_ABC(A,C):
    """Computes the gradient of matrix of Tr(ABC) with respect to B"""
    env = np.einsum("ki, jk -> ij", A, C)
    return env
def environment_tensor_two_qubit_gates(A,C):
    """Computes the gradient of matrix of Tr(ABC) with respect to B"""
    env = np.einsum("ki, jk -> ij", A, C)
    return env

def environment_tensor_simple_brickwall(even_1, odd_1, even_2, ref, layer, pos):
    """
    computes the environment tensor for a simple brickwall circuit
    --||--------||--|   |--
    --||--||----||--|   |--
          ||        |   |
    --||--||----||--|   |--
    --||--------||--|   |--
    computes the gradient at layer 'layer' and position 'pos'
    """
    ref_copy = ref.copy()
    # compute gradient of middle layer. 
    if layer == 1: 
        if pos == 0:
            # right env
            ref_copy = np.einsum("abnm, nmcdefgh->abcdefgh", even_2[0], ref_copy) #contract two-qubit gate to the right
            ref_copy = np.einsum("cdnm, abnmefgh -> abcdefgh", even_2[1], ref_copy) 
            # left env
            ref_copy = np.einsum("nmef, abcdnmgh -> abcdefgh", even_1[0], ref_copy)
            ref_copy = np.einsum("nmgh, abcdefnm -> abcdefgh", even_1[1], ref_copy)
            # upper and lower env
            
            ref_copy = np.einsum('iabjicdj->abcd', ref_copy)
    return ref_copy

    
    





