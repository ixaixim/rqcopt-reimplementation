import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_loss_series(base_dir: Path, filename: str, label: str, color: str):
    data = np.load(base_dir / filename)
    loss = np.asarray(data["loss"])
    num_gates = int(data["num_gates"])
    return {
        "label": label,
        "loss": loss,
        "num_gates": num_gates,
        "color": color,
    }


def sweep_boundaries(n_gates: int, n_iters: int):
    """
    Return a list of iteration indices for vlines.
    A vline is placed every `n_gates * 4` iterations.
    """
    period = n_gates * 4
    positions = np.arange(period, n_iters + 1, step=period)

    return positions - 1


def loss_after_sweep(loss: np.ndarray, num_gates: int):
    period = num_gates * 4
    if period == 0:
        return np.array([]), np.array([])
    sweep_indices = sweep_boundaries(num_gates, len(loss))
    if len(sweep_indices) == 0:
        return np.array([]), np.array([])
    sweep_numbers = np.arange(1, len(sweep_indices) + 1)
    return sweep_numbers, loss[sweep_indices]


base_dir = Path(__file__).resolve().parents[1] / "data"

datasets = [
    load_loss_series(base_dir, "loss_vanilla_circ.npz", "Vanilla", "C0"),
    load_loss_series(base_dir, "loss_weyl_circ.npz", "Weyl", "C3"),
    load_loss_series(base_dir, "loss_weyl_abs_circ.npz", "Weyl (absorbed)", "C1"),
]

start_iteration = 100

# Plot loss after each gate update.
fig, ax = plt.subplots(figsize=(6, 4))

for entry in datasets:
    loss = entry["loss"]
    iters = np.arange(len(loss))
    num_gates = entry["num_gates"]
    label = f"{entry['label']} – {num_gates} gates"
    color = entry["color"]

    ax.plot(iters[start_iteration:], loss[start_iteration:], label=label, lw=1.6, color=color)

    for boundary in sweep_boundaries(num_gates, len(loss)):
        if boundary >= start_iteration:
            ax.axvline(boundary, ls="--", lw=0.8, color=color, alpha=0.2)

ax.set_xlim(left=start_iteration)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_yscale("log")
ax.set_title("Local-SVD optimisation")
ax.legend()
plt.tight_layout()

output_dir = Path(__file__).resolve().parent
plt.savefig(output_dir / "weyl_optimizer_comparison.png")

# Plot loss after each full circuit sweep.
fig_sweep, ax_sweep = plt.subplots(figsize=(6, 4))

for entry in datasets:
    sweep_count, sweep_loss = loss_after_sweep(entry["loss"], entry["num_gates"])
    if len(sweep_count) == 0:
        continue
    ax_sweep.plot(sweep_count, sweep_loss, marker="o", label=entry["label"], lw=1.6, color=entry["color"])

ax_sweep.set_xlabel("Sweep")
ax_sweep.set_ylabel("Loss")
ax_sweep.set_yscale("log")
ax_sweep.set_title("Local-SVD optimisation (per sweep)")
ax_sweep.legend()
plt.tight_layout()

plt.savefig(output_dir / "weyl_optimizer_comparison_per_sweep.png")
