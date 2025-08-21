import rqcopt_mpo.jax_config

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
    is_normalized: bool = False

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
                   norm=self.norm,
                   is_normalized=self.is_normalized)
    
    def dagger(self) -> "MPO":
        """
        Return the Hermitian conjugate of the MPO.

        For every site tensor W  (l, p_out, p_in, r) we
        1) complex-conjugate the data,
        2) swap the two physical legs  p_out ↔ p_in.

        The order of sites along the chain is left unchanged, because
        the qubits themselves stay in the same order – only bra/ket
        roles swap.  
        
        Canonical flags are reset because daggering exchanges
        “left-canonical” with “right-canonical”.
        """
        dag_tensors: List[jnp.ndarray] = [
            # conj()   – complex conjugate
            # transpose – swap axes 1 ↔ 2  (p_out, p_in)
            t.conj().transpose(0, 2, 1, 3)
            for t in self.tensors
        ]

        return MPO(
            tensors=dag_tensors,
            is_left_canonical=self.is_right_canonical,
            is_right_canonical=self.is_left_canonical,
            norm=self.norm, 
            is_normalized=self.is_normalized
        )
    
    def conjugate(self) -> "MPO":
        """
        Return a new MPO where every tensor element is complex-conjugated,
        but the leg ordering, canonical flags, and norm are unchanged 
        (up to conjugation of norm if it was complex).
        """
        # apply conjugation to each site tensor
        conj_tensors = [t.conj() for t in self.tensors]

        # conjugate the stored norm if present (otherwise leave as None)
        conj_norm = None
        if self.norm is not None:
            # if norm might be complex, conjugate it; 
            # otherwise this leaves real norms untouched
            conj_norm = np.conj(self.norm)

        return MPO(
            tensors=conj_tensors,
            is_left_canonical=self.is_left_canonical,
            is_right_canonical=self.is_right_canonical,
            norm=conj_norm,
            is_normalized=self.is_normalized
        )


    def to_matrix(self) -> np.ndarray:
        """
        Contracts the MPO tensors to form the full operator matrix.

        The MPO tensor convention is assumed to be (left_bond, phys_out, phys_in, right_bond).
        The resulting matrix will have shape (physical_dim_out**n_sites, physical_dim_in**n_sites).
        """
        current_op = self.tensors[0]

        for i in range(1, self.n_sites):
            next_tensor = self.tensors[i]
            # Contract the last axis of current_op (right_bond)
            # with the first axis of next_tensor (left_bond)
            current_op = jnp.tensordot(current_op, next_tensor, axes=(-1,0))

        if current_op.shape[0] != 1 or current_op.shape[-1] != 1:
            print(f"Warning: Expected trivial boundary bonds of size 1 after full contraction, "
                f"got shape {current_op.shape}. Resulting matrix might be incorrect if MPO wasn't properly terminated.")

        full_tensor = jnp.squeeze(current_op, axis=(0, -1))
        # The shape of full_tensor is now (p_out_0, p_in_0, p_out_1, p_in_1, ..., p_out_{N-1}, p_in_{N-1})
        # We need to permute it to (p_out_0, ..., p_out_{N-1}, p_in_0, ..., p_in_{N-1})

        total_physical_indices = 2*self.n_sites
        output_indices_positions = list(range(0, total_physical_indices, 2))
        input_indices_positions = list(range(1, total_physical_indices, 2))

        permutation = tuple(output_indices_positions + input_indices_positions)
        permuted_tensor = jnp.transpose(full_tensor, axes=permutation)
                # Now reshape to the final matrix form:
        # (prod(p_out_i), prod(p_in_i))
        final_rows = self.physical_dim_out ** self.n_sites
        final_cols = self.physical_dim_in ** self.n_sites

        tensor_to_matrix = permuted_tensor.reshape(final_rows, final_cols)
        return tensor_to_matrix

    def left_canonicalize(self, normalize: bool = False) -> Optional[float]:
        """
        Performs left-to-right QR sweep for left-canonicalization.
        Modifies the MPO tensors in place.
        Returns the norm if normalize is False, otherwise None.
        """
        if self.is_left_canonical:
             print("MPO is already left-canonical.")
             return None if normalize else self.norm
             
        current_tensors = self.tensors 

        for i in range(self.n_sites - 1):
            mpo = current_tensors[i]
            shape = mpo.shape  # (l, p_out, p_in, r) 
            
            # Reshape for QR: group (l, p_out, p_in) and keep r
            left_dims_prod = np.prod(shape[:-1])
            mpo_matrix = mpo.reshape((left_dims_prod, shape[-1]))
            
            Q, R = jnp.linalg.qr(mpo_matrix, mode='reduced') # Use jax.numpy QR. Returns a matrix m x k, where k is the rank of the original matrix. 
            
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

        
        # NOTE: norm can also be complex valued (see README.txt 30.05.2025). Need to correct by absorbing phase.
        # Fixing the gauge so that R is real and positive.
        # suggested 
        phi = R_last[0,0] / jnp.abs(R_last[0,0])
        Q_last = Q_last * phi
        R_last = R_last * jnp.conj(phi)
        final_norm = R_last[0,0]
        # Update last tensor
        if normalize:
            current_tensors[-1] = Q_last.reshape(shape_last[:-1] + (Q_last.shape[-1],))
            self.norm = 1.0
            self.is_normalized = True
        else:
            self.norm = final_norm.real # imaginary part is zero, store norm as single float.
            self.is_normalized = False

        self.is_left_canonical = True
        self.is_right_canonical = False 
        
        print(f"MPO left-canonicalized. Fixed Gauge Norm: {self.norm}")
        return None if normalize else self.norm

    def right_canonicalize(self, normalize: bool = False) -> Optional[float]:
        if self.is_right_canonical:
             print("MPO is already right-canonical.")
             return None if normalize else self.norm

        current_tensors = self.tensors
        # Iterate from the second-to-last site down to the first site (0-indexed)
        for i in range(self.n_sites - 1, 0, -1):
            mpo = current_tensors[i]
            shape = mpo.shape  # (l, p_out, p_in, r)

            # Reshape for QR to isolate the left bond dimension 'l'.
            # We want M = L @ Q_core, where L is absorbed to the left.
            # A = M_matrix (l, prod(p_out, p_in, r))
            # A.T = Q_prime @ R_prime (QR decomposition)
            # A = R_prime.T @ Q_prime.T
            # So, L = R_prime.T and Q_core = Q_prime.T


            right_dims_prod = np.prod(shape[1:])
            mpo_matrix_T = mpo.reshape((shape[0], right_dims_prod)).T
            # mpo_matrix_T has shape (prod(p_out,p_in,r), l)

            Q_prime, R_prime = jnp.linalg.qr(mpo_matrix_T, mode='reduced')
            # Q_prime shape: (prod(p_out,p_in,r), new_left_bond_dim)
            # R_prime shape: (new_left_bond_dim, l)

            # Q_core is the main part of the new tensor at site i
            Q_core = Q_prime.T # shape: (new_left_bond_dim, prod(p_out,p_in,r))
            new_left_bond_dim = Q_core.shape[0]
            current_tensors[i] = Q_core.reshape((new_left_bond_dim,) + shape[1:])

            # L_to_absorb is R_prime.T, to be absorbed by the tensor to its left (i-1)
            L_to_absorb = R_prime.T # shape: (l, new_left_bond_dim)

            prev_mpo = current_tensors[i-1] # shape: (l', p_out', p_in', l)
            # merged_mpo has shape (l', p_out', p_in', new_left_bond_dim)
            merged_mpo = jnp.einsum('...k,kl->...l', prev_mpo, L_to_absorb, optimize=True)
            current_tensors[i-1] = merged_mpo

        # Handle the first tensor (site 0)
        mpo_first = current_tensors[0]
        shape_first = mpo_first.shape # (l, p_out, p_in, r) -> l should be 1 if canonical

        right_dims_prod_first = np.prod(shape_first[1:])
        mpo_matrix_T_first = mpo_first.reshape((shape_first[0], right_dims_prod_first)).T

        Q_prime_first, R_prime_first = jnp.linalg.qr(mpo_matrix_T_first, mode='reduced')
        
        
        # R_prime_first shape: (new_l_first, shape_first[0])
        # For a well-formed MPO, shape_first[0] should be 1.
        # So R_prime_first is effectively new_l_firstx1.
        L_first_matrix = R_prime_first.T # shape (shape_first[0], new_l_first)
        Q_first_core = Q_prime_first.T   # shape (new_l_first, right_dims_prod_first)

        # Fixing the gauge so that L_first_matrix[0,0] is real and positive.
        # M = L Q. We want L'[0,0] to be real.
        # L' = L * conj(phi), Q' = Q * phi. Then L'Q' = LQ.
        phi = L_first_matrix[0,0] / jnp.abs(L_first_matrix[0,0]) # phase factor
        L_first_phased = L_first_matrix * jnp.conj(phi) # Make L_first_phased[0,0] real
        Q_first_core_phased = Q_first_core * phi        # Absorb phase into Q_first_core

        final_norm = L_first_phased[0,0]

        # Update first tensor ONLY if normalize is true
        if normalize:
            current_tensors[0] = Q_first_core_phased.reshape(
                (Q_first_core_phased.shape[0],) + shape_first[1:]
            )
            self.norm = 1.0
            self.is_normalized = True
        else:
            self.norm = final_norm.real 
            self.is_normalized = False
        # If not normalizing, current_tensors[0] effectively remains L_first_phased @ Q_first_core_phased
        # which is the same as L_first_matrix @ Q_first_core, i.e., the original mpo_first (after loop contractions).

        self.is_right_canonical = True
        self.is_left_canonical = False # Cannot be both unless it's a scalar MPO (1 site)
        
        print(f"MPO right-canonicalized. Fixed Gauge Norm: {self.norm}")
        return None if normalize else self.norm

    def _invalidate_norm(self) -> None:
            """Call whenever you modify tensors without re-normalizing."""
            self.is_normalized = False

    def normalize(self) -> float:
        """
        If the MPO is left-canonical (resp. right-canonical) it
        extracts the global Frobenius norm from the *right-most*
        (resp. *left-most*) tensor, stores the resulting unit-norm
        tensor back into ``self.tensors`` and returns the old norm.

        The routine does **not** change the canonical flags – it only
        adjusts the stored norm.  It raises an error if the MPO is
        not currently in either canonical form.
        """
        # -----------------------------------
        #  Normalise a left-canonical MPO
        # -----------------------------------
        if self.is_left_canonical and not self.is_right_canonical:
            T = self.tensors[-1]                       # (ℓ, p_out, p_in, r)
            l_prod = int(np.prod(T.shape[:-1]))        # merge all but last leg
            T_mat = T.reshape((l_prod, T.shape[-1]))   #  (l_prod, r)
            Q, R = jnp.linalg.qr(T_mat, mode="reduced")

            # fix the overall phase so that R[0,0] ∈ ℝ₊
            phi  = R[0, 0] / jnp.abs(R[0, 0])
            Q  *= phi
            R  *= jnp.conj(phi)

            old_norm = float(R[0, 0].real)                 # scalar ‖M‖

            # write back the *unit* tensor
            self.tensors[-1] = Q.reshape(T.shape[:-1] + (Q.shape[-1],))
            self.norm = 1.0  
            self.is_normalized = True
            return old_norm

        # -----------------------------------
        #  Normalise a right-canonical MPO
        # -----------------------------------
        if self.is_right_canonical and not self.is_left_canonical:
            T = self.tensors[0]                        # (ℓ, p_out, p_in, r)
            r_prod = int(np.prod(T.shape[1:]))         # merge all but first leg
            T_mat_T = T.reshape((T.shape[0], r_prod)).T  #  (r_prod, ℓ)

            Qp, Rp = jnp.linalg.qr(T_mat_T, mode="reduced")
            L  = Rp.T                                  # (ℓ, k)  – left factor
            Q  = Qp.T                                  # (k, r_prod) – core

            # fix the overall phase so that L[0,0] ∈ ℝ₊
            phi  = L[0, 0] / jnp.abs(L[0, 0])
            Q  *= phi
            L  *= jnp.conj(phi)

            old_norm = float(L[0, 0].real)                 # scalar ‖M‖

            # write back the *unit* tensor
            self.tensors[0] = Q.reshape((Q.shape[0],) + T.shape[1:])
            self.norm = 1.0 # TODO: set to 1 since normalized.
            return old_norm

        # Neither flag (or both) set → ambiguous state
        raise ValueError(
            "MPO must be left- or right-canonical before calling `normalize()`."
        )

    # -----------------------------------------------------------------
    #  Pretty–print the MPO’s structure
    # -----------------------------------------------------------------
    def print_tensors(
        self,
        max_sites: int = 20,
        indent: str = "  ",
        show_bonds: bool = True,
        show_flags: bool = True,
    ) -> None:
        """
        Nicely prints the MPO layout.

        Parameters
        ----------
        max_sites : int, optional
            Maximum number of sites to list explicitly.  Beyond that an
            ellipsis line is shown to avoid flooding the console.
        indent : str, optional
            Indentation used for each site line.
        show_bonds : bool, optional
            If *True*, also prints the left→right bond dimensions per site.
        show_flags : bool, optional
            If *True*, prints canonicalisation flags & stored norm.
        """
        header = f"MPO  (n_sites = {self.n_sites},  p_out = {self.physical_dim_out}, p_in = {self.physical_dim_in})"
        if show_flags:
            header += (
                f"   [left-can: {self.is_left_canonical},  "
                f"right-can: {self.is_right_canonical},  "
                f"normalized: {self.is_normalized},  "
                f"norm: {self.norm}]"
            )
        print(header)

        if self.n_sites == 0:
            print(f"{indent}(empty)")
            return

        # Decide how many we will show
        n_show = min(self.n_sites, max_sites)
        for site in range(n_show):
            shape = tuple(self.tensors[site].shape)
            msg = f"{indent}site {site:>3}: shape {shape}"
            if show_bonds and len(shape) == 4:
                l, _, _, r = shape
                msg += f"   bonds: {l} → {r}"
            print(msg)

        if n_show < self.n_sites:
            print(f"{indent}… ({self.n_sites - n_show} more sites)")
