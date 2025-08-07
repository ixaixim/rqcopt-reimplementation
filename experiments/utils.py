from pathlib import Path
from typing import Optional
import numpy as np

def save_data_npz(base_dir: Path, name: str, loss: list[float], num_gates: Optional[int] = None):
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fn = data_dir / f"{name}.npz"

    # data to save
    data = {
        "loss": np.array(loss),
    }
    if num_gates is not None:
        data["num_gates"] = np.array(num_gates)

    np.savez(fn, **data)
    print(f"Saved NPZ to {fn}")
    # np.save(fn, floats)


