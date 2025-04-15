from dataclasses import dataclass, field
import numpy as np
import jax.numpy as jnp # Or stick to numpy if preferred
from typing import List, Optional, Tuple

@dataclass
class MPO:
    """Represents a Matrix Product Operator."""
    tensors: List[jnp.ndarray] # List of MPO tensors (l, p_out, p_in, r) or similar convention
    # field(init=False): class variable of type int, not included as a parameter in the __init()__ method, i.e. not set by user at initialization.
    n_sites: int = field(init=False)
    physical_dim_out: int = field(init=False) # Assuming uniform physical dim 
    physical_dim_in: int = field(init=False) # Assuming uniform physical dim
    
    # Metadata
    is_left_canonical: bool = False
    is_right_canonical: bool = False
    norm: Optional[float] = None # Store norm after canonicalization

    def __post_init__(self):
        if not self.tensors:
            raise ValueError("MPO tensors list cannot be empty.")
        self.n_sites = len(self.tensors)
        # Infer physical dimensions from the first tensor (assuming uniformity)
        # Adjust indices based on your convention, e.g., (left, phys_out, phys_in, right)
        shape = self.tensors[0].shape 
        if len(shape) != 4:
             print(f"Warning: Expected 4D MPO tensors, got shape {shape}. Adjust class if convention differs.")
            #  # Handle potential boundary vector cases if necessary
            #  self.physical_dim_out = shape[1] if len(shape)>1 else 1
            #  self.physical_dim_in = shape[2] if len(shape)>2 else 1
        else:
             self.physical_dim_out = shape[1] 
             self.physical_dim_in = shape[2] 
        # Basic shape consistency check (optional but good)
        for i in range(self.n_sites - 1):
            if self.tensors[i].shape[-1] != self.tensors[i+1].shape[0]:
                raise ValueError(f"Bond dimension mismatch between MPO tensor {i} and {i+1}")

    def __len__(self) -> int:
        return self.n_sites

    def __getitem__(self, index) -> jnp.ndarray:
        return self.tensors[index]

    def copy(self) -> 'MPO':
        # Ensure deep copy of tensors
        new_tensors = [t.copy() for t in self.tensors]
        return MPO(tensors=new_tensors, 
                   is_left_canonical=self.is_left_canonical,
                   is_right_canonical=self.is_right_canonical,
                   norm=self.norm)
    
    def left_canonicalize(self, normalize: bool = False) -> Optional[float]:
        """
        Performs left-to-right QR sweep for left-canonicalization.
        Modifies the MPO tensors in place.
        Returns the norm if normalize is False, otherwise None.
        """
        if self.is_left_canonical and (self.norm == 1.0 or not normalize) :
             print("MPO is already left-canonical.")
             return None if normalize else self.norm
             
        current_tensors = self.tensors # Work directly on the list

        for i in range(self.n_sites - 1):
            mpo = current_tensors[i]
            shape = mpo.shape  # (l, p_out, p_in, r) 
            
            # Reshape for QR: group (l, p_out, p_in) and keep r
            left_dims_prod = np.prod(shape[:-1])
            mpo_matrix = mpo.reshape((left_dims_prod, shape[-1]))
            
            Q, R = jnp.linalg.qr(mpo_matrix, mode='reduced') # Use jax.numpy QR
            
            # Reshape Q back and update tensor
            new_right_bond_dim = Q.shape[-1]
            current_tensors[i] = Q.reshape(shape[:-1] + (new_right_bond_dim,))
            
            next_mpo = current_tensors[i+1]
            merged_mpo = jnp.einsum('ij,j...->i...', R, next_mpo, optimize=True) # ellipsis '...' means "all remaining trailing dimensions"
            current_tensors[i+1] = merged_mpo # Shape (new_r, p_out', p_in', r')
        
        # Handle the last tensor
        mpo_last = current_tensors[-1]
        shape_last = mpo_last.shape
        left_dims_prod_last = np.prod(shape_last[:-1])
        mpo_matrix_last = mpo_last.reshape((left_dims_prod_last, shape_last[-1]))

        Q_last, R_last = jnp.linalg.qr(mpo_matrix_last, mode='reduced')

        # Extract norm from R_last (should be 1x1)
        final_norm = R_last.reshape(-1)[0]

        # Ensure positive norm convention (optional but common)
        # Note: absirbing the norm is performed for stability: the first element of R is the norm since R will have dimensions (left_bond_dim, 1) due to the last MPO tensor having trivial right bond dimension. R is an upper triangular matrix, therefore only first element of R is nonzero. 
        if final_norm < 0:
            final_norm = -final_norm
            Q_last = -Q_last

        self.norm = float(final_norm.real) # Store the norm

        # Update last tensor
        if normalize:
            current_tensors[-1] = Q_last.reshape(shape_last[:-1] + (Q_last.shape[-1],))
            self.norm = 1.0
        else:
            # Absorb norm into the last tensor if not normalizing
            # Q_last already has shape (..., 1), multiply by scalar norm
            current_tensors[-1] = (Q_last * final_norm).reshape(shape_last[:-1] + (Q_last.shape[-1],))

        self.is_left_canonical = True
        self.is_right_canonical = False # Cannot be both unless it's a scalar
        
        print(f"MPO left-canonicalized. Norm: {self.norm}")
        return None if normalize else self.norm
