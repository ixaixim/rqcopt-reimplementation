from __future__ import annotations
import rqcopt_mpo.jax_config

from typing import Dict, List, Tuple, Any, Iterator, Callable
import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

from rqcopt_mpo.optimization.parametrized_adam.utils import _rot_from_generator, _paulis_1q, _paulis_2q
# NOTE: pytree technique might be slower, since we are making each parameter a leaf, i.e. an array. 
        # It might be smarter to have a whole array for at least each layer, thereby grouping the parameters.

# ---- Type aliases: the parameter PyTree and a parallel "meta" tree ----
# Each leaf is a 1D jnp.ndarray of angles for that gate (possibly length 0 for fixed gates).
ParamsTree = Dict[int, Dict[int, jnp.ndarray]]     # layer_index -> gate_index -> theta (1D)
MetaTree   = Dict[int, Dict[int, dict]]            # same keys; small dict with metadata per gate

def _flatten_gate_params_to_vec(params_tuple: Tuple[Any, ...],
                                dtype=jnp.float64) -> Tuple[jnp.ndarray, List[int]]:
    """
    Pack Gate.params (tuple of scalars/arrays) into a single 1D vector (jnp.ndarray).
    Returns (theta, splits) where 'splits' records the raveled sizes of each element.

    If the gate is not parameterized (empty tuple), returns a length-0 vector and splits=[].
    """
    if not params_tuple:
        return jnp.zeros((0,), dtype=dtype), []

    chunks: List[jnp.ndarray] = []
    splits: List[int] = []

    for p in params_tuple:
        a = jnp.asarray(p, dtype=dtype).ravel()
        chunks.append(a)
        splits.append(int(a.size))

    theta = jnp.concatenate(chunks) if chunks else jnp.zeros((0,), dtype=dtype)
    return theta, splits

def _unflatten_vec_to_gate_params(theta: jnp.ndarray, splits: List[int]) -> Tuple[jnp.ndarray, ...]:
    """
    Inverse of _flatten_gate_params_to_vec when you only need a flat tuple of 1D arrays.
    (You can post-process each piece to your preferred shape if needed.)
    """
    if not splits:
        return tuple()
    out: List[jnp.ndarray] = []
    start = 0
    for s in splits:
        out.append(theta[start:start + s])
        start += s
    return tuple(out)

# ------------------------- building the PyTrees -------------------------

def extract_params_tree(circ) -> Tuple[ParamsTree, MetaTree]:
    """
    Build a params PyTree for *all* gates in the circuit and a parallel meta tree.
    - Keys: params[layer_idx][gate_idx] = 1D jnp.ndarray of angles (len 0 if no params)
    - Meta keeps: {'arity': 1 or 2, 'name': str, 'qubits': tuple, 'splits': List[int]}

    This is stable because it uses layer.iterate_gates() which sorts gates spatially.
    """
    circ.sort_layers()
    params_tree: ParamsTree = {}
    meta_tree: MetaTree = {}

    for layer in circ.layers:
        layer_params: Dict[int, jnp.ndarray] = {}
        layer_meta: Dict[int, dict] = {}

        for gi, g in enumerate(layer.iterate_gates()):
            theta, splits = _flatten_gate_params_to_vec(g.params, dtype=jnp.float64)
            layer_params[gi] = theta
            layer_meta[gi] = {
                "arity": len(g.qubits),
                "name": g.name,
                "qubits": tuple(g.qubits),
                "splits": splits,  # for potential reconstruction
            }

        params_tree[layer.layer_index] = layer_params
        meta_tree[layer.layer_index] = layer_meta

    return params_tree, meta_tree

def tree_like_zeros(params_tree: ParamsTree) -> ParamsTree:
    """Make a zeros tree with the same structure and dtypes as params_tree."""
    return jtu.tree_map(lambda x: jnp.zeros_like(x), params_tree)

# ---------------------------- useful masks -----------------------------

# def masks_by_arity(meta_tree: MetaTree) -> Tuple[ParamsTree, ParamsTree]:
#     """
#     Build (mask_1q, mask_2q), same PyTree structure as params_tree.
#     Each leaf is an array of 1s or 0s with the same shape as the parameter vector for that gate.
#     Non-parameterized gates (len 0) just get empty arrays.
#     """
#     mask_1q: ParamsTree = {}
#     mask_2q: ParamsTree = {}

#     for layer_idx, gates_meta in meta_tree.items():
#         m1_layer: Dict[int, jnp.ndarray] = {}
#         m2_layer: Dict[int, jnp.ndarray] = {}
#         for gi, meta in gates_meta.items():
#             size = int(np.sum(meta["splits"])) if meta["splits"] else 0
#             if meta["arity"] == 1:
#                 m1_layer[gi] = jnp.ones((size,), dtype=jnp.float64)
#                 m2_layer[gi] = jnp.zeros((size,), dtype=jnp.float64)
#             elif meta["arity"] == 2:
#                 m1_layer[gi] = jnp.zeros((size,), dtype=jnp.float64)
#                 m2_layer[gi] = jnp.ones((size,), dtype=jnp.float64)
#             else:  # future-proof: gates with other arity
#                 m1_layer[gi] = jnp.zeros((size,), dtype=jnp.float64)
#                 m2_layer[gi] = jnp.zeros((size,), dtype=jnp.float64)
#         mask_1q[layer_idx] = m1_layer
#         mask_2q[layer_idx] = m2_layer

#     return mask_1q, mask_2q

# --------------------- optional: reconstruction helpers ----------------

def rotate_rx(theta: jnp.ndarray, meta: dict, dtype=jnp.complex128) -> jnp.ndarray:
    # theta shape (1,)
    scale = float(meta.get("exp_scale", 0.5))
    X, _, _, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], X, scale, dtype)        # 2x2
    return jnp.array(U)

def rotate_ry(theta: jnp.ndarray, meta: dict, dtype=jnp.complex128) -> jnp.ndarray:
    # theta shape (1,)
    scale = float(meta.get("exp_scale", 0.5))
    _, Y, _, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], Y, scale, dtype)        # 2x2
    return jnp.array(U)

def rotate_rz(theta: jnp.ndarray, meta: dict, dtype=jnp.complex128) -> jnp.ndarray:
    # theta shape (1,)
    scale = float(meta.get("exp_scale", 0.5))
    _, _, Z, _ = _paulis_1q(dtype)

    U = _rot_from_generator(theta[0], Z, scale, dtype)        # 2x2
    return jnp.array(U)

def rotate_rxx_ryy_rzz(theta: jnp.ndarray, meta: dict, dtype=jnp.complex128) -> jnp.ndarray:
    # theta shape (1,)
    if theta.shape[0] != 3:
        raise ValueError(f"Expected theta shape (3,), got {theta.shape}")
    scale = float(meta.get("exp_scale", 1.))

    XX, YY, ZZ, I4 = _paulis_2q(dtype)
    Ua = _rot_from_generator(theta[0], XX, scale, dtype)
    Ub = _rot_from_generator(theta[1], YY, scale, dtype)
    Uc = _rot_from_generator(theta[2], ZZ, scale, dtype)
    U  = Ua @ Ub @ Uc

    return jnp.array(U)


def update_gate_params_inplace(circ, params_tree: ParamsTree, meta_tree: MetaTree) -> None:
    """
    Mutates the circuit's Gate.params to reflect the given params_tree.
    Keeps each gate's params as a tuple of 1D arrays (you can reshape later per gate type).
    NOTE: If you prefer *functional* updates, use a 'rebuild_circuit' function instead.
    """
    rotation_matrix: Dict[str, Callable[[jnp.ndarray, dict], jnp.ndarray]] = {}
    
    rotation_matrix['RX'] = rotate_rx
    rotation_matrix['RY'] = rotate_ry
    rotation_matrix['RZ'] = rotate_rz
    rotation_matrix['Exp(XX+YY+ZZ)'] = rotate_rxx_ryy_rzz

    circ.sort_layers()
    dtype = getattr(circ, "dtype", jnp.complex128)
    for layer in circ.layers:
        layer_idx = layer.layer_index
        gi = 0
        for gi, g in enumerate(layer.iterate_gates()):
            theta = params_tree[layer_idx][gi]
            meta = meta_tree[layer_idx][gi]
            splits = meta["splits"]
            g.params = _unflatten_vec_to_gate_params(theta, splits)
            fn = rotation_matrix[meta['name']]
            g.matrix = fn(theta, meta, dtype) 
