import time
import jax
import jax.numpy as jnp
import numpy as np
import os
import gc
from typing import List, Tuple, Dict

from rqcopt_mpo.circuit.circuit_builder import generate_random_circuit
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.gradient import sweeping_euclidean_gradient_bottom_up, sweeping_euclidean_gradient_top_down
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit
from rqcopt_mpo.tensor_network.core_ops_helpers import hs_inner_product_from_mpo

# Ensure we use complex128 for precision
jax.config.update("jax_enable_x64", True)

def compute_loss_dense(circuit: Circuit, target_matrix: jnp.ndarray):
    """Loss function: Tr(V^\dagger U(circuit)) using dense matrices"""
    U = circuit.to_matrix()
    return jnp.trace(jnp.conjugate(target_matrix).T @ U)

def compute_loss_mpo(circuit: Circuit, target_mpo: 'MPO'):
    """Loss function: Tr(V^\dagger U(circuit)) using MPOs"""
    # Note: max_bondim=None and svd_cutoff=0.0 ensures we don't truncate 
    # based on values, which helps JAX keep static shapes.
    U_mpo = circuit_to_mpo(circuit, max_bondim=128, svd_cutoff=0.0)
    return hs_inner_product_from_mpo(target_mpo, U_mpo)

def get_jax_dense_grads(circuit: Circuit, target_matrix: jnp.ndarray):
    """Compute gradients using jax.grad on dense matrix"""
    grad_fn = jax.grad(lambda c: compute_loss_dense(c, target_matrix), holomorphic=True)
    grads_tree = grad_fn(circuit)
    grads_ordered = []
    for layer in grads_tree.layers:
        for gate in layer.iterate_gates(reverse=False):
            grads_ordered.append(gate.matrix)
    return grads_ordered

def get_jax_mpo_grads(circuit: Circuit, target_mpo: 'MPO'):
    """Compute gradients using jax.grad on MPO representation"""
    grad_fn = jax.grad(lambda c: compute_loss_mpo(c, target_mpo), holomorphic=True)
    grads_tree = grad_fn(circuit)
    grads_ordered = []
    for layer in grads_tree.layers:
        for gate in layer.iterate_gates(reverse=False):
            grads_ordered.append(gate.matrix)
    return grads_ordered

def run_experiment(n_sites: int, n_layers: int, circuit_type: str = "random"):
    print(f"\n--- Experiment: n_sites={n_sites}, n_layers={n_layers}, type={circuit_type} ---")
    
    # 1. Setup
    if circuit_type == "random":
        circuit = generate_random_circuit(n_sites=n_sites, n_layers=n_layers, seed=42, dtype=jnp.complex128)
    else: # trotter
        circuit = trotterized_hardware_friendly_xyz_circuit(
            n_sites=n_sites, Jx=1.0, Jy=0.5, Jz=0.2, hx=0.3, dt=0.1, reps=max(1, n_layers // 4), order=2, dtype=jnp.complex128
        )
        n_layers = len(circuit.layers)
        print(f"Adjusted n_layers for Trotter: {n_layers}")
    
    # Target unitary
    target_circuit = generate_random_circuit(n_sites=n_sites, n_layers=min(n_layers, 2), seed=43, dtype=jnp.complex128)
    target_mpo = circuit_to_mpo(target_circuit)
    target_mpo.left_canonicalize()

    # Target matrix (only for small systems)
    target_matrix = None
    if n_sites <= 11:
        target_matrix = target_circuit.to_matrix()

    # 2. JAX Dense Gradient
    jax_dense_grads = None
    jax_dense_time = float('inf')
    
    if target_matrix is not None:
        gc.collect()
        try:
            # Warm up
            _ = get_jax_dense_grads(circuit, target_matrix)
            start_time = time.time()
            jax_dense_grads = get_jax_dense_grads(circuit, target_matrix)
            jax_dense_grads = [g.block_until_ready() for g in jax_dense_grads]
            jax_dense_time = time.time() - start_time
            print(f"JAX-Dense grad time: {jax_dense_time:.4f}s")
        except Exception as e:
            print(f"JAX-Dense grad failed: {e}")

    # 3. JAX MPO Gradient
    jax_mpo_grads = None
    jax_mpo_time = float('inf')
    
    # Run JAX MPO if n_sites is manageable (e.g. <= 20)
    if n_sites <= 16:
        gc.collect()
        try:
            # Warm up
            _ = get_jax_mpo_grads(circuit, target_mpo)
            start_time = time.time()
            jax_mpo_grads = get_jax_mpo_grads(circuit, target_mpo)
            jax_mpo_grads = [g.block_until_ready() for g in jax_mpo_grads]
            jax_mpo_time = time.time() - start_time
            print(f"JAX-MPO grad time: {jax_mpo_time:.4f}s")
        except Exception as e:
            print(f"JAX-MPO grad failed: {e}")
    else:
        print("Skipping JAX-MPO grad due to large n_sites.")

    # 4. TN Sweeping Gradient (Bottom-Up)
    gc.collect()
    # Warm up
    _, _, _ = sweeping_euclidean_gradient_bottom_up(
        circuit=circuit, mpo_ref=target_mpo, max_bondim_env=128, svd_cutoff=1e-12
    )
    start_time = time.time()
    tn_loss_bu, tn_grads_bu, _ = sweeping_euclidean_gradient_bottom_up(
        circuit=circuit, mpo_ref=target_mpo, max_bondim_env=128, svd_cutoff=1e-12
    )
    for g in tn_grads_bu:
        g.block_until_ready()
    tn_time_bu = time.time() - start_time
    print(f"TN grad (Bottom-Up) time: {tn_time_bu:.4f}s")

    # 5. Compare
    results_comp = {}
    if jax_dense_grads is not None:
        max_diff_bu = max(jnp.max(jnp.abs(jg.reshape(tg.shape) - tg)) for jg, tg in zip(jax_dense_grads, tn_grads_bu))
        print(f"Max grad diff (JAX-Dense vs TN-BU): {max_diff_bu:.2e}")
        results_comp["dense_vs_tn"] = max_diff_bu < 1e-8

    if jax_mpo_grads is not None:
        max_diff_mpo_tn = max(jnp.max(jnp.abs(jg.reshape(tg.shape) - tg)) for jg, tg in zip(jax_mpo_grads, tn_grads_bu))
        print(f"Max grad diff (JAX-MPO vs TN-BU): {max_diff_mpo_tn:.2e}")
        results_comp["mpo_vs_tn"] = max_diff_mpo_tn < 1e-8

    return {
        "n_sites": n_sites,
        "n_layers": n_layers,
        "jax_dense_time": jax_dense_time,
        "jax_mpo_time": jax_mpo_time,
        "tn_time_bu": tn_time_bu,
        "results_comp": results_comp
    }

if __name__ == "__main__":
    results = []
    # Small cases for correctness
    for ns in [4, 6, 8, 10]:
        results.append(run_experiment(ns, 4, "random"))
    
    # Mid/Large cases
    for ns in [12, 16, 20]:
        results.append(run_experiment(ns, 4, "random"))
        
    # Trotterized circuit
    results.append(run_experiment(16, 20, "trotter"))

    print("\nSummary Table (Times in seconds):")
    print("-" * 85)
    print(f"{'n_sites':>7} | {'n_layers':>8} | {'JAX-Dense':>10} | {'JAX-MPO':>10} | {'TN-BU':>10} | {'Match'}")
    print("-" * 85)
    for r in results:
        jd_t = f"{r['jax_dense_time']:10.4f}" if r['jax_dense_time'] != float('inf') else f"{'N/A':>10}"
        jm_t = f"{r['jax_mpo_time']:10.4f}" if r['jax_mpo_time'] != float('inf') else f"{'N/A':>10}"
        match = "YES" if all(r["results_comp"].values()) else "NO" if r["results_comp"] else "N/A"
        print(f"{r['n_sites']:7d} | {r['n_layers']:8d} | {jd_t} | {jm_t} | {r['tn_time_bu']:10.4f} | {match}")
    print("-" * 85)
