import sys
import os
import csv
import numpy as np
import jax.numpy as jnp
import jax

# Ensure rqcopt_mpo is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rqcopt_mpo.jax_config
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_xyz_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.optimization.utils import overlap_to_loss
from rqcopt_mpo.tensor_network.core_ops_helpers import hs_inner_product_from_mpo



# -----------------------------------------------------------------------------
# Main Experiment
# -----------------------------------------------------------------------------

def run_experiment():
    print("Starting Trotter Error Comparison Experiment...")

    # --- Parameters ---
    # Reference settings
    # Using 10 reps as requested
    REF_ORDER = 4
    REF_REPS = 20 
    REF_NORMALIZE = True
    
    # Physics settings (XXX Heisenberg)
    N_SITES = 6
    JX = 0.0
    JY = 0.0
    JZ = 1.0
    HX = 0.75
    HY = 0.0
    HZ = 0.6
    TOTAL_TIME = 2.0
    loss_name = "HST" # can be either HST or frobenius
    max_bondim_ref = 64


    # Sweep settings
    orders_list = [1, 2, 4]
    reps_dict = {
        1: np.arange(1, 26), 
        2: np.arange(1, 25), 
        4: np.arange(1, 6)
    }
    
    # orders_list = [2, 4]
    # reps_dict = {
    #     2: np.arange(25, 34),
    #     4: np.arange(9, 12)
    # }

    # Output file
    output_csv = os.path.join(os.path.dirname(__file__), "trotter_comparison_results.csv")

    # --- 1. Construct Reference MPO ---
    print(f"\nNumber of sites: {N_SITES}")
    print(f"Building Reference MPO (Order={REF_ORDER}, Reps={REF_REPS}, T={TOTAL_TIME})...")
    dt_ref = TOTAL_TIME / REF_REPS
    
    circ_ref = trotterized_hardware_friendly_xyz_circuit(
        n_sites=N_SITES,
        Jx=JX, Jy=JY, Jz=JZ, hx=HX, hy=HY, hz=HZ,
        order=REF_ORDER,
        dt=dt_ref,
        reps=REF_REPS,
        method='suzuki',
        collapse=True
    )

    mpo_ref = circuit_to_mpo(circ_ref, max_bondim=max_bondim_ref, svd_cutoff=0.0)
    mpo_ref.left_canonicalize(normalize=REF_NORMALIZE) 
    
    # --- 2. Run Sweep ---
    results = []
    
    print(f"{ 'Order':<6} | { 'Reps':<5} | { 'dt':<8} | { 'Overlap (Real)':<15} | { f'{loss_name} Loss':<15}")
    print("-----------------------------------------------------------------")

    for order in orders_list:
        reps_range = reps_dict[order]
        for reps in reps_range:
            dt = TOTAL_TIME / reps
            
            # Build approximate circuit
            circ_approx = trotterized_hardware_friendly_xyz_circuit(
                n_sites=N_SITES,
                Jx=JX, Jy=JY, Jz=JZ, hx=HX, hy=HY, hz=HZ,
                order=order,
                dt=dt,
                reps=int(reps),
                method='suzuki', # 'yoshida' or 'suzuki
                collapse=True
            )
            
            mpo_approx = circuit_to_mpo(circ_approx, max_bondim=128, svd_cutoff=1e-12)
            mpo_approx.left_canonicalize(normalize=False)
            
            # Compute Overlap Tr(U_ref^dag U_approx)
            overlap = hs_inner_product_from_mpo(mpo_ref, mpo_approx)
            
            # Compute HST Loss
            # We treat both as unitaries on N_SITES qubits, so they are "unnormalized" MPOs 
            # (norm^2 = 2^N). Passing normalize=False accounts for this.
            hst_loss = overlap_to_loss(overlap, kind=loss_name, n_sites=N_SITES, normalize=REF_NORMALIZE)
            
            print(f"{order:<6} | {reps:<5} | {dt:<8.4f} | {np.real(overlap):<15.4e} | {float(hst_loss):<15.4e}")
            
            results.append({
                "order": order,
                "reps": reps,
                "dt": dt,
                "overlap_real": np.real(overlap),
                "overlap_imag": np.imag(overlap),
                "hst_loss": float(hst_loss)
            })

    # --- 3. Save Results ---
    print(f"\nSaving results to {output_csv}...")
    
    keys = results[0].keys()
    
    with open(output_csv, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print("Done.")

if __name__ == "__main__":
    run_experiment()