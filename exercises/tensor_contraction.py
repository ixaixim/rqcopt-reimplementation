import numpy as np

def absorb_to_the_right(two_qubit_gate, big_tensor, i, N, preserve_open=False):
    """
    Absorbs a two-qubit gate into a 2^N x 2^N operator acting on N qubits to the right.
    
    Parameters:
      two_qubit_gate : numpy array of shape (4,4)
          The two-qubit gate to absorb.
      big_tensor : numpy array of shape (2**N, 2**N)
          The operator acting on N qubits.
      i : int
          The positional index (0-indexed) where the gate acts on qubits i and i+1.
      N : int
          Total number of qubits.
          
    Returns:
      new_tensor : numpy array of shape (2**N, 2**N)
          The updated operator after absorbing the two-qubit gate.
    """
    
    # 1. Reshape the two-qubit gate into a 4-index tensor of shape (2,2,2,2).
    #    Convention: gate_tensor[a, b, c, d] with a,b = new indices; c,d = indices to be contracted.
    gate_tensor = two_qubit_gate.reshape(2, 2, 2, 2)
    
    # 2. Reshape the big operator into a 2N-index tensor.
    #    We assume the first N indices correspond to the input ("left") part and the last N indices to the output.
    big_shape = (2,) * (2 * N)
    big_tensor_reshaped = big_tensor.reshape(big_shape)
    
    # 3. Contract the two-qubit gate with the big operator.
    #    We contract the gate's indices 2 and 3 with the big tensor’s input indices at positions i and i+1.
    contracted = np.tensordot(gate_tensor, big_tensor_reshaped, axes=([2, 3], [i, i+1]))
    
    # 4. Rearrange the resulting tensor.
    #    After tensordot, the ordering is:
    #      - First two axes: the gate’s non-contracted (new) indices.
    #      - Then the remaining input indices (those not at positions i and i+1) followed by all output indices.
    #
    #    We need to insert the two new indices into the input part at positions i and i+1.
    #    Using np.moveaxis, we move axes 0 and 1 to positions i and i+1.
    new_tensor = np.moveaxis(contracted, [0, 1], [i, i+1])
    
    if preserve_open:
        # Do not collapse the free indices.
        return new_tensor
    else:
        # Fully contract back to a matrix.
        return new_tensor.reshape(2**N, 2**N)



def absorb_to_the_left(two_qubit_gate, big_tensor, i, N, preserve_open=False):
    gate_tensor = two_qubit_gate.reshape(2, 2, 2, 2)
    big_shape = (2,) * (2 * N)
    big_tensor_reshaped = big_tensor.reshape(big_shape)
    contracted = np.tensordot(gate_tensor, big_tensor_reshaped, axes=([0, 1], [N+i, N+i+1]))
    new_tensor = np.moveaxis(contracted, [0, 1], [N+i, N+i+1])
    if preserve_open:
        return new_tensor
    else:
        return new_tensor.reshape(2**N, 2**N)
    
    return new_tensor
