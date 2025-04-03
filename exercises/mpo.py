import numpy as np
# construct an MPO from matrix

def mpo_from_matrix(M, N):
    """
    Convert a matrix M of size 2^N x 2^N to an MPO representation.
    """
    mpo = []
    # Express M as a tensor with 2*N physical indices (each of dimension 2)
    new_shape = [2 for _ in range(2 * N)]
    R = M.reshape(*new_shape)

    # First local tensor: intended shape (2, 2, 1, D)
    # We permute the indices so that the two physical indices for this tensor come first.
    axes_order = [0, N] + [j for j in range(1, 2 * N) if j != N]
    R_perm = R.transpose(axes_order)

    # Combine the first two physical indices into the "row" dimension.
    row_dim = 2 * 2
    num_physical_indexes = 2 * N - 2
    col_dim = 2 ** (num_physical_indexes)
    R_perm = R_perm.reshape(row_dim, col_dim)
    
    # Perform SVD on the reshaped matrix.
    U, S, Vh = np.linalg.svd(R_perm, full_matrices=False)
    D = S.shape[0]  # bond dimension from the SVD
    
    # Reshape U into the first MPO tensor with shape (2, 2, 1, D)
    local_tensor = U.reshape(2, 2, 1, D)
    mpo.append(local_tensor)
    
    # Form the remainder R by multiplying the singular values into Vh.
    R = np.dot(np.diag(S), Vh)
    # Reshape R to have the remaining physical indices and the new bond dimension.
    new_shape = [2 for _ in range(2 * N - 2)] + [D]
    R = R.reshape(*new_shape)

    # middle local tensors:TODO
    for i in range(1,N-1):
        # permute axes to collect the physical indexes of the i-th tensor
        axes_order = [0, N-i] + [len(R.shape)-1] + [j for j in range(1, 2*(N-i)) if j!=(N-i)] 
        # TODO:put the bond dim axis after the two physical legs, to ensure that the indices that we want to group together are contiguous in the current memory layout. 
        R_perm = R.transpose(axes_order)
        # reshape to matrix 
        row_dim = 2*2*D
        num_physical_indexes = 2*(N-1-i) 
        col_dim = 2 ** (num_physical_indexes)
        R_perm = R_perm.reshape(row_dim, col_dim)

        # Perform SVD.
        U, S, Vh = np.linalg.svd(R_perm, full_matrices=False)
        DD = S.shape[0]  # new right bond dimension

        local_tensor = U.reshape(2,2,D,DD)
        mpo.append(local_tensor)
        print(f"Tensor {i}: Left bond dim: {D}, Right bond dim: {DD}")

        # Absorb S into Vh and reshape the remainder.
        R = np.dot(np.diag(S), Vh)
        new_shape = [2 for _ in range(num_physical_indexes)] + [DD]
        R = R.reshape(*new_shape)
        D = DD  # update bond dimension for next iteration

    # Last local tensor:
    # R should now be of shape (2, 2, D, 1).
    # We force this reshape and append it.
    local_tensor = R.reshape(2, 2, D, 1)
    mpo.append(local_tensor)
    return mpo


if __name__ == "__main__":
        # Choose a system size, e.g., N = 3 sites => matrix dimension 2^3 x 2^3 = 8 x 8.
        N = 4
        dim = 2 ** N
        # Create a random matrix.
        M = np.random.randn(dim, dim)
        # Obtain the MPO representation from the matrix.
        mpo = mpo_from_matrix(M, N)



