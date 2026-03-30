import rqcopt_mpo.jax_config
import jax.numpy as jnp
from pathlib import Path
import sys
from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_xyz_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.mpo.mpo_dataclass import MPO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
from experiments.utils import get_reference_path

def create_heisenberg_reference(
    n_sites: int,
    Jx: float, Jy: float, Jz: float,
    hx: float = 0.0, hy: float = 0.0, hz: float = 0.0,
    t: float = 1.0,
    reps: int = 20,
    order: int = 4,
    dtype = jnp.complex128,
    max_bondim_ref: int = 64,
    canonicalize: str = "left",
    normalize: bool = False,
) -> MPO:
    """
    Generate a reference MPO for a Heisenberg-style Hamiltonian evolution.
    """
    dt = t / reps
    
    target_circ = trotterized_xyz_circuit(
        n_sites=n_sites,
        Jx=Jx, Jy=Jy, Jz=Jz,
        hx=hx, hy=hy, hz=hz,
        order=order,
        method='suzuki',
        dt=dt,
        reps=reps,
        dtype=dtype
    )
    
    target_mpo = circuit_to_mpo(target_circ, max_bondim=max_bondim_ref, svd_cutoff=0.0)
    
    if canonicalize == "left":
        target_mpo.left_canonicalize(normalize=normalize)
    elif canonicalize == "right":
        target_mpo.right_canonicalize(normalize=normalize)
        
    return target_mpo
if __name__ == "__main__":
    # Riemannian experiment reference
    n_sites = 20
    Jx = 0.0
    Jy = 0.0
    Jz = 1.0
    hx = 0.75
    hy = 0.0
    hz = 0.6
    t = 2.0
    reps = 20
    order = 4

    mpo = create_heisenberg_reference(
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz, t=t, reps=reps, order=order,
        normalize=True
    )
    
    # Save to a central location
    ref_base_dir = REPO_ROOT / "experiments"
    path = get_reference_path(
        base_dir=ref_base_dir,
        n_sites=n_sites, Jx=Jx, Jy=Jy, Jz=Jz, hx=hx, hy=hy, hz=hz, t=t, reps=reps, order=order
    )
    
    mpo.save_json(path)
    print(f"Saved unified reference to {path}")
