from pathlib import Path
import numpy as np

def save_data_npz(base_dir: Path, name: str, loss: list[float], num_gates: int):
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    fn = data_dir / f"{name}.npz"
    np.savez(fn,
            loss=np.array(loss),
            num_gates=np.array(num_gates))
    print(f"Saved NPZ to {fn}")
    # np.save(fn, floats)


