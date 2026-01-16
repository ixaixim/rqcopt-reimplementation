import rqcopt_mpo.jax_config
import os
import csv
import uuid
import jax.numpy as jnp
import numpy as np
from rqcopt_mpo.optimization.adam_utils import make_early_stop
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.circuit.trotter.trotter_hardware_friendly import trotterized_hardware_friendly_xyz_circuit
from rqcopt_mpo.circuit.rzz_ising_decompose.rzz_circuit_builder_hw_friendly import rzz_decompose_ising_circuit
from rqcopt_mpo.optimization.rzz_ising_optimizer.rzz_ising_optimizer_hw_friendly import optimize

def run_experiment():
    # File to save results
    # Using absolute path relative to project root or current working dir
    output_dir = "experiments/rzz_ising_experiment/data"
    os.makedirs(output_dir, exist_ok=True)
    loss_dir = os.path.join(output_dir, "loss_histories")
    os.makedirs(loss_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "sweep_results.csv")
    
    # Define columns
    fieldnames = [
        "id", "n", "J", "D", "hx", "hz", "t",  # System
        "ansatz_reps", "ansatz_order", "ansatz_dt", # Circuit (Ansatz)
        "target_reps", "target_order", "target_dt", # Circuit (Target)
        "lr", "max_steps", "max_bondim_env", "svd_cutoff", # Optimization
        "final_loss", "initial_circuit_2q_layers" # Results
    ]
    
    # Initialize CSV if needed
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

    print(f"Saving results to: {csv_file}")

    # System Parameters
    n_sites = 20
    J = 0.0
    D = 1.0
    hx = 0.75
    hz = 0.6
    t = 2.0
    dtype = jnp.complex128

    if J != 0 or hx == 0 or hz == 0:
         print("Warning: J should be 0 and fields non-zero for TFIM setup if strictly following original script constraints.")

    # Target Circuit Parameters (Fixed Reference)
    target_order = 4
    target_reps = 20
    target_dt = t / target_reps
    
    # Build Target MPO 
    print("Building Target MPO...")
    target_circ = trotterized_hardware_friendly_xyz_circuit(    
        n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz,
        order=target_order, dt=target_dt, reps=target_reps, collapse=True,
        dtype=dtype
    )
    target_mpo = circuit_to_mpo(target_circ)
    print(f"Target MPO built. 2Q Layers: {target_circ.num_2q_layers}")

    # Optimization Hyperparameters
    max_steps = 2000
    lr = 1e-4
    betas = (0.9, 0.999)
    eps = 1e-8
    clip_grad_norm = None
    max_bondim_env = 128
    svd_cutoff = 0.0

    # Sweep Parameters
    orders_list = [1,2,4] # Only order 2 is used in this example
    reps_dict = {1: np.arange(1, 26), 2: np.arange(1, 25), 4: np.arange(1, 9)}

    for order in orders_list:
        for reps in reps_dict[order]:
            dt = t / reps
            
            print(f"\n--- Starting Optimization: Order={order}, Reps={reps} ---")
            
            try:
                # Build Ansatz Circuit
                initial_circuit = trotterized_hardware_friendly_xyz_circuit(    
                    n_sites=n_sites, Jx=J, Jy=J, Jz=D, hx=hx, hz=hz, 
                    order=order, dt=dt, reps=reps, collapse=True,
                    dtype=dtype
                )
                
                num_2q_layers = initial_circuit.num_2q_layers
                
                # Decompose
                new_circ = rzz_decompose_ising_circuit(initial_circuit, order)
                
                early_stop = make_early_stop(patience=10, min_delta=1e-12, target_loss=1e-13)

                # Optimize
                optimized_circ, loss_history = optimize(
                    new_circ,
                    target_mpo,
                    max_steps=max_steps,
                    max_bondim_env=max_bondim_env,
                    svd_cutoff=svd_cutoff,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                    clip_grad_norm=clip_grad_norm,
                    use_ad=False,
                    callback=early_stop,
                )
                
                final_loss = min(loss_history) if loss_history else float('nan')
                
                # Generate unique ID and save loss history
                experiment_id = str(uuid.uuid4())
                np.save(os.path.join(loss_dir, f"{experiment_id}.npy"), loss_history)
                
                # Log Result
                result = {
                    "id": experiment_id,
                    "n": n_sites, "J": J, "D": D, "hx": hx, "hz": hz, "t": t,
                    "ansatz_reps": reps, "ansatz_order": order, "ansatz_dt": dt,
                    "target_reps": target_reps, "target_order": target_order, "target_dt": target_dt,
                    "lr": lr, "max_steps": max_steps, "max_bondim_env": max_bondim_env, "svd_cutoff": svd_cutoff,
                    "final_loss": final_loss, "initial_circuit_2q_layers": num_2q_layers
                }
                
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
                    
                print(f"Saved result: Loss={final_loss:.6e}")
                
            except Exception as e:
                print(f"Error for Order={order}, Reps={reps}: {e}")
                # Log failure to CSV if desired, or just skip
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
