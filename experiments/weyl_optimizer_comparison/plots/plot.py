import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# plotting
base_dir = Path(__file__).resolve().parents[1] / "data"
fn = base_dir / "loss_vanilla_circ.npz"
data = np.load(fn)
loss_vanilla = data["loss"]        
num_gates_vanilla_circ  = int(data["num_gates"])    

fn = base_dir / "loss_weyl_circ.npz"
data = np.load(fn)
loss_weyl = data["loss"]        
num_gates_weyl_circ  = int(data["num_gates"])    


loss_vanilla = np.asarray(loss_vanilla)
loss_weyl    = np.asarray(loss_weyl)

iters_vanilla = np.arange(len(loss_vanilla))
iters_weyl    = np.arange(len(loss_weyl))

def sweep_boundaries(n_gates: int, n_iters: int):
    """
    Return a list of iteration indices for vlines.
    A vline is placed every `n_gates * 4` iterations.
    """
    period = n_gates * 4
    positions = np.arange(period, n_iters + 1, step=period)

    return positions-1

vlines_vanilla = sweep_boundaries(num_gates_vanilla_circ, len(loss_vanilla))
vlines_weyl    = sweep_boundaries(num_gates_weyl_circ,    len(loss_weyl))

fig, ax = plt.subplots(figsize=(6, 4))

# filter from iteration 100 onwards
start_iteration = 100

ax.plot(iters_vanilla[start_iteration:], loss_vanilla[start_iteration:],
        label=f"Vanilla – {num_gates_vanilla_circ} gates",
        lw=1.6)
ax.plot(iters_weyl[start_iteration:], loss_weyl[start_iteration:],
        label=f"Weyl – {num_gates_weyl_circ} gates",
        lw=1.6)

for i in vlines_vanilla:
    if i >= start_iteration:
        ax.axvline(i, ls="--", lw=0.8, color="grey", alpha=0.35)

for i in vlines_weyl:
    if i >= start_iteration:
        ax.axvline(i, ls="--", lw=0.8, color="red", alpha=0.25)

# You can optionally still set the x-limit to ensure it starts exactly at 100
ax.set_xlim(left=start_iteration)


ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_yscale("log")                 # comment out if you want linear scale
ax.set_title("Local-SVD optimisation")
ax.legend()
plt.tight_layout()

fn = Path(__file__).resolve().parent / "weyl_optimizer_comparison.png"
plt.savefig(fn)


