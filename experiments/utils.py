from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rqcopt_mpo.circuit.trotter.trotter_circuit_builder import trotterized_heisenberg_circuit
from rqcopt_mpo.mpo.mpo_builder import circuit_to_mpo
from rqcopt_mpo.mpo.mpo_dataclass import MPO
from rqcopt_mpo.circuit.circuit_dataclasses import Circuit


def load_experiment_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON configuration file describing a trotterized experiment.
    Returns a plain dict so callers can further customize parameters.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _dtype_from_name(name: str) -> Any:
    name = name.lower()
    if name in ("complex128", "c128", "complex"):
        return jnp.complex128
    if name in ("complex64", "c64"):
        return jnp.complex64
    raise ValueError(f"Unsupported dtype '{name}' in config")


def _build_heisenberg_trotter_circuit(
    problem_cfg: dict[str, Any],
    trotter_cfg: dict[str, Any],
) -> Circuit:
    """
    Build a trotterized Heisenberg circuit from two config subsections:
    'problem': static Hamiltonian parameters, and a trotter-specific block.
    """
    dtype = _dtype_from_name(problem_cfg.get("dtype", "complex128"))
    reps = trotter_cfg["reps"]
    t_total = problem_cfg["time"]
    dt = t_total / reps

    return trotterized_heisenberg_circuit(
        n_sites=problem_cfg["n_sites"],
        J=problem_cfg["J"],
        D=problem_cfg["D"],
        h=problem_cfg.get("h", 0.0),
        order=trotter_cfg["order"],
        dt=dt,
        reps=reps,
        dtype=dtype,
    )


def build_target_mpo_from_config(cfg: dict[str, Any]) -> MPO:
    """
    Convenience wrapper that reads the 'problem' + 'target_mpo' sections
    and returns a canonicalized MPO ready for optimization pipelines.
    """
    problem_cfg = cfg["problem"]
    target_cfg = cfg["target_mpo"]

    target_circuit = _build_heisenberg_trotter_circuit(problem_cfg, target_cfg)
    mpo = circuit_to_mpo(target_circuit)

    canonicalization = target_cfg.get("canonicalize", "left")
    if canonicalization == "left":
        mpo.left_canonicalize(normalize=target_cfg.get("normalize", False))
    elif canonicalization == "right":
        mpo.right_canonicalize(normalize=target_cfg.get("normalize", False))
    elif canonicalization not in (None, "none"):
        raise ValueError(f"Unknown canonicalization directive '{canonicalization}'")

    return mpo


def build_initial_circuit_from_config(cfg: dict[str, Any]) -> Circuit:
    """
    Build the variational circuit specified by the 'init_circuit' block.
    """
    problem_cfg = cfg["problem"]
    init_cfg = cfg["init_circuit"]
    return _build_heisenberg_trotter_circuit(problem_cfg, init_cfg)


@dataclass(frozen=True)
class OptimizationSetup:
    name: str
    max_steps: int
    max_bondim_env: int
    svd_cutoff: float
    optimizer_params: dict[str, Any]


def load_optimization_setup(cfg: dict[str, Any]) -> OptimizationSetup:
    """
    Extract optimizer metadata (name + hyperparameters) from config.
    """
    block = cfg.get("optimization")
    if not block:
        raise ValueError("Missing 'optimization' block in config.")

    return OptimizationSetup(
        name=block["optimizer"],
        max_steps=block["max_steps"],
        max_bondim_env=block["max_bondim_env"],
        svd_cutoff=block.get("svd_cutoff", 1e-12),
        optimizer_params=block.get("optimizer_params", {}),
    )


def save_data_npz(
    base_dir: Path,
    name: str,
    loss: list[float],
    num_gates: Optional[int] = None,
    method: Optional[str] = None
):
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fn = data_dir / f"{name}.npz"

    # data to save
    data = {
        "loss": np.array(loss, dtype=float),
    }
    if num_gates is not None:
        data["num_gates"] = np.array(num_gates, dtype=int)
    if method is not None:
        data["method"] = np.array(method)  # npz requires array-like

    np.savez(fn, **data)
    print(f"Saved NPZ to {fn} (keys: {list(data.keys())})")
    # NOTE: for later runs, consider adding a time stamp
    # np.save(fn, floats)


def save_run_outputs(
    base_dir: Path,
    run_name: str,
    circuit: Circuit,
    loss: list[float],
    num_gates: Optional[int] = None,
    method: Optional[str] = None,
) -> dict[str, Path]:
    """
    Save loss history and circuit JSON into a dedicated run directory.

    Returns:
        Mapping of artifact names to their paths.
    """
    run_dir = base_dir / "data" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loss_data = {"loss": np.array(loss, dtype=float)}
    if num_gates is not None:
        loss_data["num_gates"] = np.array(num_gates, dtype=int)
    if method is not None:
        loss_data["method"] = np.array(method)

    loss_path = run_dir / "loss.npz"
    circuit_path = run_dir / "circuit.json"

    np.savez(loss_path, **loss_data)
    circuit.save_json(circuit_path)

    print(f"Saved loss to {loss_path} and circuit to {circuit_path}")
    return {"loss": loss_path, "circuit": circuit_path}

def load_all_npz(data_dir: Path):
    """
    Load all .npz files from a given data/ directory.
    Returns:
        dict: {filename_stem: dict_of_arrays}
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    results = {}
    for fn in data_dir.glob("*.npz"):
        with np.load(fn, allow_pickle=True) as data:
            results[fn.stem] = {key: data[key] for key in data.files}

    return results

def plot_all_losses(
    all_data: dict,
    out_name: str,
    out_dir: Optional[Path] = None,
    dpi: int = 150,
    fmt: str = "png",
) -> Path:
    """
    Plot loss vs. iteration for all runs and save to disk.

    Args:
        all_data: {name: {"loss": np.array, "method": np.array or str, ...}}
        out_name: filename without extension (e.g. "loss_plot")
        out_dir: directory to save into. If None, uses the directory of this file.
        dpi: image DPI for savefig.
        fmt: output format (e.g. 'png', 'pdf', 'svg').

    Returns:
        Path to the saved figure.
    """
    if out_dir is None:
        try:
            out_dir = Path(__file__).resolve().parent
        except NameError:
            out_dir = Path.cwd()

    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{out_name}.{fmt}"

    # --- Create figure and explicit axis ---
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Plot all loss curves ---
    for name, content in all_data.items():
        loss = content.get("loss")
        method = content.get("method")

        if loss is None:
            print(f"Skipping {name}: no 'loss' key")
            continue

        loss = np.asarray(loss).reshape(-1)

        # Normalize method to string
        if isinstance(method, np.ndarray):
            method = method.item() if method.shape == () else str(method)
        if method is None:
            method = name

        x = np.arange(1, len(loss) + 1)
        ax.plot(x, loss, label=str(method))

    # --- Log scale + ticks ---
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=tuple(range(2, 10)))
    )
    ax.tick_params(axis='y', which='minor', length=4, color='gray')
    ax.tick_params(axis='y', which='major', length=7)

    # --- Labels, title, grid ---
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs. Iteration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(fn, dpi=dpi, format=fmt)
    plt.close(fig)

    print(f"Saved plot to {fn}")
    return fn
