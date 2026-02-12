
import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from rqcopt_mpo.hamiltonian.xyz_model import XYZModel
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import (
    trotterized_xyz_layers
)
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit

def test_suzuki_vs_yoshida():
    Jx, Jy, Jz = 1.0, 0.8, 0.5
    hx, hy, hz = 0.1, 0.2, 0.3
    n_sites      = 4
    total_time_t = 0.5

    # Build XYZ Hamiltonian
    model        = XYZModel(n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, 
                            hx=hx, hy=hy, hz=hz, dtype=jnp.complex128)
    H_full       = model.build_hamiltonian_matrix()

    # Exact evolution operator U(t) = exp(-1j * t * H)
    U_exact = expm(-1j * total_time_t * H_full)

    def trotter_unitary(order: int, method: str, delta_t: float, n_steps: int) -> jnp.ndarray:
        layers = trotterized_xyz_layers(
            n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz,
            hx=hx, hy=hy, hz=hz,
            order=order, method=method, dt=delta_t, reps=n_steps,
            dtype=jnp.complex128
        )
        circ = Circuit(
            n_sites=n_sites,
            layers=layers,
        )
        return circ.to_matrix()

    def spectral_error(U_target: jnp.ndarray, U_approx: jnp.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(U_target - U_approx), ord=2))

    n_steps_list = [5, 10, 20]
    
    yoshida_errors = []
    suzuki_errors = []
    
    for n_steps in n_steps_list:
        dt = total_time_t / n_steps
        
        U_yoshida = trotter_unitary(4, "yoshida", dt, n_steps)
        err_yoshida = spectral_error(U_exact, U_yoshida)
        yoshida_errors.append(err_yoshida)
        
        U_suzuki = trotter_unitary(4, "suzuki", dt, n_steps)
        err_suzuki = spectral_error(U_exact, U_suzuki)
        suzuki_errors.append(err_suzuki)
        
        print(f"n_steps={n_steps}, dt={dt:.4f}")
        print(f"  Yoshida error: {err_yoshida:.2e}")
        print(f"  Suzuki error:  {err_suzuki:.2e}")

    # Check for 4th order scaling: error should drop by ~2^4 = 16 when doubling n_steps
    for i in range(len(n_steps_list) - 1):
        ratio_yoshida = yoshida_errors[i] / yoshida_errors[i+1]
        ratio_suzuki = suzuki_errors[i] / suzuki_errors[i+1]
        print(f"Ratio (n={n_steps_list[i]} to n={n_steps_list[i+1]}):")
        print(f"  Yoshida: {ratio_yoshida:.2f} (expected ~16)")
        print(f"  Suzuki:  {ratio_suzuki:.2f} (expected ~16)")
        
        assert ratio_yoshida > 10, f"Yoshida does not show 4th order scaling: {ratio_yoshida}"
        assert ratio_suzuki > 10, f"Suzuki does not show 4th order scaling: {ratio_suzuki}"

if __name__ == "__main__":
    test_suzuki_vs_yoshida()
