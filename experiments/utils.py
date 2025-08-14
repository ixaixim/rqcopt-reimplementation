from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


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
        # Folder where this python file lives
        try:
            out_dir = Path(__file__).resolve().parent
        except NameError:
            # Fallback for interactive sessions
            out_dir = Path.cwd()

    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{out_name}.{fmt}"

    plt.figure(figsize=(8, 5))

    for name, content in all_data.items():
        loss = content.get("loss")
        method = content.get("method")

        if loss is None:
            print(f"Skipping {name}: no 'loss' key")
            continue

        # Convert to 1D float array if possible
        loss = np.asarray(loss).reshape(-1)

        # method may be np.ndarray with dtype=object or 0-d array; normalize to str
        if isinstance(method, np.ndarray):
            method = method.item() if method.shape == () else str(method)
        if method is None:
            method = name

        x = np.arange(1, len(loss) + 1)
        plt.plot(x, loss, label=str(method))

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs. Iteration")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(fn, dpi=dpi, format=fmt)
    plt.close()

    print(f"Saved plot to {fn}")
    return fn



