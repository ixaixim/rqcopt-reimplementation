import argparse
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.utils import (  # noqa: E402
    build_initial_circuit_from_config,
    build_target_mpo_from_config,
    load_experiment_config,
    load_optimization_setup,
    OptimizationSetup,
)
from rqcopt_mpo.circuit.weyl_decomposition.weyl_circuit_builder import (  # noqa: E402
    weyl_decompose_circuit,
    absorb_single_qubit_layers,
)
from rqcopt_mpo.optimization.weyl_optimizer.weyl_abs_optimizer import (  # noqa: E402
    optimize as optimize_weyl_absorption,
)

from rqcopt_mpo.optimization.riemannian_adam.optimizer import (  # noqa: E402
    optimize as optimize_riemannian_adam,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Weyl absorption experiment from config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/heisenberg/sample_order4_r10.json"),
        help="Path to experiment configuration JSON.",
    )
    return parser.parse_args()


def _as_tuple(sequence: Sequence[float] | float | None) -> tuple[float, float]:
    if sequence is None:
        return (0.9, 0.999)
    if isinstance(sequence, (float, int)):
        return (float(sequence), float(sequence))
    return tuple(float(x) for x in sequence)  # type: ignore[arg-type]


def run_experiment(cfg_path: Path) -> None:
    cfg = load_experiment_config(cfg_path)
    target_mpo = build_target_mpo_from_config(cfg)
    init_circ = build_initial_circuit_from_config(cfg)
    opt_setup = load_optimization_setup(cfg)


    print(f"Loaded config '{cfg.get('name', cfg_path.stem)}'")
    print(f"Target MPO: n_sites={target_mpo.n_sites}, normalized={target_mpo.is_normalized}")
    # print)Ä
    print(f"Initial circuit layers={init_circ.num_layers}, gates={init_circ.num_gates}")
    print(f"Optimizer: {opt_setup.name}")

    if opt_setup.name == "weyl_abs":
        circuit = _prepare_weyl_abs_circuit(init_circ)
        loss_history = _run_weyl_absorption(circuit, target_mpo, opt_setup)
    elif opt_setup.name == "riemannian_adam":
        loss_history = _run_riemannian_adam(init_circ, target_mpo, opt_setup)
    else:
        raise NotImplementedError(f"Unsupported optimizer '{opt_setup.name}'.")

    print(f"Finished optimization, {len(loss_history)} steps.")
    if loss_history:
        print(f"Final loss: {loss_history[-1]}")

def _prepare_weyl_abs_circuit(init_circ):
    circ = init_circ.copy()
    circ = weyl_decompose_circuit(circ, keep_global_phase=False)
    circ = absorb_single_qubit_layers(circ)
    return circ

def _run_weyl_absorption(circuit, target_mpo, opt_setup: OptimizationSetup) -> None:
    params = opt_setup.optimizer_params
    betas = _as_tuple(params.get("betas"))

    return optimize_weyl_absorption(
        circuit,
        mpo_ref=target_mpo,
        max_steps=opt_setup.max_steps,
        max_bondim_env=opt_setup.max_bondim_env,
        svd_cutoff=opt_setup.svd_cutoff,
        lr=params.get("lr", 1e-3),
        betas=betas,
        eps=params.get("eps", 1e-8),
        clip_grad_norm=params.get("clip_grad_norm"),
        bias_correction=params.get("bias_correction", True),
    )

def _run_riemannian_adam(circuit, target_mpo, opt_setup: OptimizationSetup):
    params = opt_setup.optimizer_params
    betas = _as_tuple(params.get("betas"))
    return optimize_riemannian_adam(
        circuit,
        target_mpo,
        lr=params.get("lr", 1e-3),
        betas=betas,
        eps=params.get("eps", 1e-8),
        clip_grad_norm=params.get("clip_grad_norm"),
        max_steps=opt_setup.max_steps,
        max_bondim_env=opt_setup.max_bondim_env,
        svd_cutoff=opt_setup.svd_cutoff,
        callback=None,
        init_vertical_sweep=params.get("init_vertical_sweep", "top-down"),
    )

def main() -> None:
    args = parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()
