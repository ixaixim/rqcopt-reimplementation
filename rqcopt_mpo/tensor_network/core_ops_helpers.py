import rqcopt_mpo.jax_config

import jax.numpy as jnp
from rqcopt_mpo.mpo.mpo_dataclass import MPO

def compress_SVD(u, s, v, max_bondim=128):
    '''
    Compress an MPO to a maximum bond dimension max_bondim.
    '''
    if max_bondim>=len(s): pass
    else:
        u = u[..., :max_bondim]
        v = v[:max_bondim, ...]
        s = s[:max_bondim]
    return u, s, v

def get_mpo_from_matrix(U, max_bondim=128):
    '''
    Decompose a full-rank matrix into an MPO by means of SVD.
    Returns an MPO object.
    '''
    tensor_list = []
    n = int(jnp.round(jnp.log2(jnp.shape(U)[0])))

    shape = (2,2,)*n
    A = U.reshape(shape)

    for site in range(1, n):
        n_ = n-site+1  # Particles still left to SVD
        if site==1: A_perm = jnp.moveaxis(A, n_, 1)
        elif 1<site<n: A_perm = jnp.moveaxis(A, n_+1, 2)

        shape_1 = A_perm.shape
        lim=2 if site==1 else 3  # Exception for first round
        shape_2 = (int(jnp.prod(jnp.asarray(shape_1[:lim]))),
                   int(jnp.prod(jnp.asarray(shape_1[lim:]))))
        
        B = A_perm.reshape(shape_2)

        u,s,v = jnp.linalg.svd(B, full_matrices=False)
        u, s, v = compress_SVD(u, s, v, max_bondim)
        
        D = jnp.diag(s)@v
        shape_3 = shape_1[:lim] + (u.shape[-1],)
        
        E = u.reshape(shape_3)
        tensor_list.append(E)
        shape_4 = (u.shape[-1],) + shape_1[lim:]
        F = D.reshape(shape_4)
        if site==n-1: tensor_list.append(F)
        A = F.copy()

    # Add dummy legs
    tensor_list[0] = tensor_list[0][jnp.newaxis,...]
    tensor_list[-1] = tensor_list[-1][...,jnp.newaxis]

    return MPO(tensors=tensor_list)

def hs_inner_product_from_mpo(A_mpo, B_mpo):
    """
    Compute Tr(A^dag B) where A_mpo, B_mpo are MPOs.
    Assumes matching bond dimensions at boundaries (usually 1).
    """
    # Conjugate transpose A (dagger) at the MPO level
    A_dag = A_mpo.dagger()

    # Left environment starts as (1,1)
    # L_env corresponds to contraction of left bonds.
    # shape (A_bond, B_bond)
    # Assuming the first tensor has left bond dim 1
    # Check if bond dims match roughly or if they are 1.
    # Usually for MPO, first left bond is 1.
    
    L_env = jnp.eye(1, dtype=A_mpo[0].dtype).reshape(1, 1)

    # Contract site by site.
    for i in range(len(A_dag)):
        # A_dag[i] shape: (l_a, p_out, p_in, r_a)
        # B_mpo[i] shape: (l_b, p_out, p_in, r_b)
        # L_env shape: (l_a, l_b)
        # We want to contract L_env with left legs of A_dag and B_mpo
        # and contract physical legs of A_dag and B_mpo
        # Resulting shape: (r_a, r_b)
        
        # A_dag indices: a (left), b (p_in), c (p_out), d (right)
        # B_mpo indices: e (left), c (p_out), b (p_in), f (right)
        # L_env indices: a, e
        
        # We want to contract:
        # A_dag.p_in (b) with B_mpo.p_in (b)
        # A_dag.p_out (c) with B_mpo.p_out (c)
        L_env = jnp.einsum('ae, abcd, ecbf -> df', L_env, A_dag[i], B_mpo[i], optimize=True)

    # scalar overlap
    return L_env[0, 0]
